# Proactive Rate Limiting (Fixed-Interval Pacer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a cross-process proactive rate-limiting mechanism (pacer) to avoid `429` responses by ensuring a minimum time interval between requests.

**Architecture:** A `RateLimitGuard` class will manage a shared state file in `~/.hermes/rate_limits/`. It will use `fcntl.flock` for cross-process synchronization to calculate and reserve the next available request time slot, effectively serializing the start of requests across all sessions.

**Tech Stack:** Python, `fcntl`, `json`, `time`, `threading`.

---

### Task 1: Create `agent/rate_limit_guard.py`

**Files:**
- Create: `agent/rate_limit_guard.py`
- Test: `tests/agent/test_rate_limit_guard.py`

- [ ] **Step 1: Write initial tests for RateLimitGuard**

```python
import os
import time
import shutil
import tempfile
from unittest.mock import patch
from agent.rate_limit_guard import RateLimitGuard

def test_pacer_waits_correct_interval():
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("agent.rate_limit_guard._state_dir", return_value=tmp_dir):
            guard = RateLimitGuard("test-model", rpm=60) # 1s interval
            
            start = time.time()
            guard.acquire() # first call immediate
            guard.acquire() # second call should wait ~1s
            elapsed = time.time() - start
            
            assert 0.9 <= elapsed <= 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_rate_limit_guard.py -v`
Expected: FAIL (ImportError or ModuleNotFoundError)

- [ ] **Step 3: Implement RateLimitGuard**

```python
import json
import logging
import os
import time
import fcntl
from typing import Optional

logger = logging.getLogger(__name__)

def _state_dir() -> str:
    try:
        from hermes_constants import get_hermes_home
        base = get_hermes_home()
    except ImportError:
        base = os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(base, "rate_limits")

class RateLimitGuard:
    def __init__(self, key: str, rpm: int):
        self.key = key.replace("/", "_").replace(":", "_")
        self.interval = 60.0 / rpm
        self.path = os.path.join(_state_dir(), f"{self.key}.json")

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
        # We use an auxiliary lock file to avoid issues with flock on the JSON file itself
        lock_path = self.path + ".lock"
        
        with open(lock_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            
            now = time.time()
            state = {"last_allowed_at": 0.0}
            
            if os.path.exists(self.path):
                try:
                    with open(self.path, "r") as sf:
                        state = json.load(sf)
                except (json.JSONDecodeError, ValueError):
                    pass
            
            last_allowed = state.get("last_allowed_at", 0.0)
            target_time = max(now, last_allowed + self.interval)
            
            state["last_allowed_at"] = target_time
            
            with open(self.path, "w") as sf:
                json.dump(state, sf)
                
            # Lock released on close of 'f'
        
        wait_time = target_time - now
        if wait_time > 0:
            logger.info("Rate limit pacer: waiting %.2fs for %s", wait_time, self.key)
            time.sleep(wait_time)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_rate_limit_guard.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/rate_limit_guard.py tests/agent/test_rate_limit_guard.py
git commit -m "feat: add RateLimitGuard for proactive rate limiting"
```

---

### Task 2: Update Provider Profile and NVIDIA Profile

**Files:**
- Modify: `providers/base.py`
- Modify: `plugins/model-providers/nvidia/__init__.py`

- [ ] **Step 1: Add rate_limit_rpm to ProviderProfile**

Modify `providers/base.py`:
```python
@dataclass
class ProviderProfile:
    # ...
    rate_limit_rpm: int = 0  # 0 = disabled
```

- [ ] **Step 2: Update NVIDIA profile**

Modify `plugins/model-providers/nvidia/__init__.py`:
```python
nvidia = ProviderProfile(
    # ...
    rate_limit_rpm=40,
)
```

- [ ] **Step 3: Commit**

```bash
git add providers/base.py plugins/model-providers/nvidia/__init__.py
git commit -m "feat: add rate_limit_rpm to ProviderProfile and enable for NVIDIA"
```

---

### Task 3: Integrate with AIAgent

**Files:**
- Modify: `run_agent.py`

- [ ] **Step 1: Locate _interruptible_api_call and add pacing**

Modify `run_agent.py`:
```python
    def _interruptible_api_call(self, api_kwargs: dict):
        # ...
        # BEFORE starting the thread
        try:
            from providers import get_provider_profile
            profile = get_provider_profile(self.provider)
            if profile and profile.rate_limit_rpm > 0:
                from agent.rate_limit_guard import RateLimitGuard
                RateLimitGuard(f"{self.provider}:{self.model}", profile.rate_limit_rpm).acquire()
        except Exception as e:
            logger.debug("Rate limit guard error in _interruptible_api_call: %s", e)
```

- [ ] **Step 2: Locate _interruptible_streaming_api_call and add pacing**

Modify `run_agent.py`:
```python
    def _interruptible_streaming_api_call(self, api_kwargs: dict, *, on_first_delta: callable = None):
        # ...
        # BEFORE starting the thread (or before calling _interruptible_api_call for Codex)
        try:
            from providers import get_provider_profile
            profile = get_provider_profile(self.provider)
            if profile and profile.rate_limit_rpm > 0:
                from agent.rate_limit_guard import RateLimitGuard
                RateLimitGuard(f"{self.provider}:{self.model}", profile.rate_limit_rpm).acquire()
        except Exception as e:
            logger.debug("Rate limit guard error in _interruptible_streaming_api_call: %s", e)
```

- [ ] **Step 3: Commit**

```bash
git add run_agent.py
git commit -m "feat: integrate RateLimitGuard into AIAgent"
```

---

### Task 4: Integrate with AuxiliaryClient

**Files:**
- Modify: `agent/auxiliary_client.py`

- [ ] **Step 1: Locate call_llm and acall_llm and add pacing**

Modify `agent/auxiliary_client.py`:
```python
def call_llm(...):
    # ...
    # After provider/model resolution, before client call
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(resolved_provider)
        if profile and profile.rate_limit_rpm > 0:
            from agent.rate_limit_guard import RateLimitGuard
            RateLimitGuard(f"{resolved_provider}:{final_model}", profile.rate_limit_rpm).acquire()
    except Exception:
        pass
```

- [ ] **Step 2: Commit**

```bash
git add agent/auxiliary_client.py
git commit -m "feat: integrate RateLimitGuard into AuxiliaryClient"
```
