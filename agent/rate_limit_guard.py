import json
import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

logger = logging.getLogger(__name__)

def _state_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
        base = get_hermes_home()
    except ImportError:
        base = Path(os.path.expanduser("~")) / ".hermes"
    return base / "rate_limits"

@contextmanager
def _file_lock(lock_path: Path):
    if fcntl is None and msvcrt is None:
        yield
        return

    # Ensure file exists and has content for msvcrt
    if not lock_path.exists() or lock_path.stat().st_size == 0:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("lock", encoding="utf-8")

    with open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8") as f:
        try:
            if fcntl:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            elif msvcrt:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            elif msvcrt:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass

class RateLimitGuard:
    def __init__(self, key: str, rpm: Optional[int]):
        # Robust slugification: replace non-alphanumeric (except _ and -) with _
        self.key = re.sub(r'[^a-zA-Z0-9_\-]', '_', key)
        self.rpm = rpm
        if rpm and rpm > 0:
            self.interval = 60.0 / rpm
        else:
            self.interval = 0.0
        self.path = _state_dir() / f"{self.key}.json"

    def acquire(self):
        wait_time = self._reserve_slot()
        if wait_time > 0:
            logger.info("Rate limit pacer: waiting %.2fs for %s", wait_time, self.key)
            time.sleep(wait_time)

    async def async_acquire(self):
        wait_time = self._reserve_slot()
        if wait_time > 0:
            import asyncio
            logger.info("Rate limit pacer: waiting %.2fs (async) for %s", wait_time, self.key)
            await asyncio.sleep(wait_time)

    def _reserve_slot(self) -> float:
        """
        Reserve the next available time slot in the shared state file.
        Returns the time (in seconds) the caller should wait before proceeding.
        """
        if self.interval <= 0:
            return 0.0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # We use an auxiliary lock file to avoid issues with flock on the JSON file itself
        lock_path = self.path.with_suffix(".json.lock")
        
        with _file_lock(lock_path):
            now = time.time()
            state = {"last_allowed_at": 0.0}
            
            if self.path.exists():
                try:
                    with open(self.path, "r", encoding="utf-8") as sf:
                        state = json.load(sf)
                except (json.JSONDecodeError, ValueError):
                    pass
            
            last_allowed = state.get("last_allowed_at", 0.0)
            target_time = max(now, last_allowed + self.interval)
            
            state["last_allowed_at"] = target_time
            
            with open(self.path, "w", encoding="utf-8") as sf:
                json.dump(state, sf)
        
        return target_time - now
