import os
import time
import tempfile
import concurrent.futures
import asyncio
from pathlib import Path
from unittest.mock import patch
import pytest
from agent.rate_limit_guard import RateLimitGuard

def test_pacer_waits_correct_interval():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Patch the _state_dir function to use our temp directory
        with patch("agent.rate_limit_guard._state_dir", return_value=tmp_path):
            guard = RateLimitGuard("test-model", rpm=60) # 1req/s interval
            
            start = time.monotonic()
            guard.acquire() # first call immediate
            guard.acquire() # second call should wait ~1s
            elapsed = time.monotonic() - start
            
            # Allow for some overhead/scheduler jitter, but it must be at least 0.9s
            assert 0.9 <= elapsed <= 1.5

def test_concurrent_acquire_pacing():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with patch("agent.rate_limit_guard._state_dir", return_value=tmp_path):
            # 120 RPM = 0.5s interval
            guard = RateLimitGuard("concurrent-test", rpm=120)
            
            num_threads = 4
            start = time.monotonic()
            
            def worker():
                guard.acquire()
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker) for _ in range(num_threads)]
                concurrent.futures.wait(futures)
                
            elapsed = time.monotonic() - start
            # 4 requests with 0.5s interval:
            # Thread 1: starts at T=0
            # Thread 2: starts at T=0.5
            # Thread 3: starts at T=1.0
            # Thread 4: starts at T=1.5
            # Total elapsed should be at least 1.5s
            assert elapsed >= 1.4

def test_rpm_zero_disables_pacing():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with patch("agent.rate_limit_guard._state_dir", return_value=tmp_path):
            guard = RateLimitGuard("no-pacing", rpm=0)
            
            start = time.monotonic()
            guard.acquire()
            guard.acquire()
            guard.acquire()
            elapsed = time.monotonic() - start
            
            # Should be almost instantaneous
            assert elapsed < 0.1

@pytest.mark.asyncio
async def test_async_pacer_waits_correct_interval():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with patch("agent.rate_limit_guard._state_dir", return_value=tmp_path):
            guard = RateLimitGuard("test-model-async", rpm=60) # 1s interval
            
            start = time.monotonic()
            await guard.async_acquire() # first call immediate
            await guard.async_acquire() # second call should wait ~1s
            elapsed = time.monotonic() - start
            
            # Allow some jitter
            assert 0.9 <= elapsed <= 1.5

@pytest.mark.asyncio
async def test_async_concurrent_acquire_pacing():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with patch("agent.rate_limit_guard._state_dir", return_value=tmp_path):
            guard = RateLimitGuard("concurrent-test-async", rpm=120) # 0.5s interval
            
            # Start 3 acquires concurrently
            start = time.monotonic()
            tasks = [guard.async_acquire() for _ in range(3)]
            await asyncio.gather(*tasks)
            elapsed = time.monotonic() - start
            
            # 3 requests with 0.5s interval:
            # R1: 0s
            # R2: 0.5s
            # R3: 1.0s
            # Total elapsed should be at least 1.0s
            assert 1.0 <= elapsed <= 1.5
