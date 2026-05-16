# Design Spec: Proactive Rate Limiting (Fixed-Interval Pacer)

## Goal
Implement a proactive, cross-process rate-limiting mechanism to avoid `429` responses by ensuring a minimum time interval between requests (e.g., 1.5s for 40 RPM).

## Requirements
- **Global enforcement**: Shared across all sessions via `~/.hermes/rate_limits/`.
- **Fixed Interval**: Ensures at least `60 / RPM` seconds between the *start* of any two requests.
- **Queue and delay**: Requests wait their turn to start.
- **NVIDIA Default**: 40 RPM (1.5s interval).

## Proposed Changes

### 1. New Module: `agent/rate_limit_guard.py`
The `RateLimitGuard` will act as a "pacer". It schedules the next allowed request time in a shared file.

```python
class RateLimitGuard:
    def __init__(self, key: str, rpm: int):
        self.key = _sanitize_key(key)
        self.interval = 60.0 / rpm
        self.path = _get_state_path(self.key)

    def acquire(self):
        """
        Calculates and reserves the next available time slot.
        Returns immediately after updating the shared state, 
        then sleeps locally if needed.
        """
        with open(self.path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            # Read last_allowed_time, calculate wait, update last_allowed_time
            # ...
        
        if wait_time > 0:
            time.sleep(wait_time)
```

### 2. Provider Profile Updates
Update `ProviderProfile` in `providers/base.py` to include `rate_limit_rpm`.
Update `plugins/model-providers/nvidia/__init__.py` to set `rate_limit_rpm=40`.

### 3. Integration in `AIAgent` (`run_agent.py`)
In `_interruptible_api_call` and `_interruptible_streaming_api_call`, before starting the API thread.

### 4. Integration in `AuxiliaryClient` (`agent/auxiliary_client.py`)
In `call_llm` and `acall_llm`, before the client call.
