import time
import asyncio
from enum import Enum

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self):
        self._failure_count: int = 0
        self._success_count_half_open: int = 0
        self._state: CircuitState = CircuitState.CLOSED
        self._last_failure_time: float = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= settings.CB_RECOVERY_TIMEOUT:
                self._state = CircuitState.HALF_OPEN
                self._success_count_half_open = 0
        return self._state

    def record_success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._success_count_half_open += 1
            if self._success_count_half_open >= settings.CB_HALF_OPEN_MAX_CALLS:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        else:
            self._failure_count = 0

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= settings.CB_FAILURE_THRESHOLD:
            self._state = CircuitState.OPEN


class CircuitOpenError(Exception):
    pass


_breakers: dict[int, CircuitBreaker] = {}


def get_breaker(model_id: int) -> CircuitBreaker:
    if model_id not in _breakers:
        _breakers[model_id] = CircuitBreaker()
    return _breakers[model_id]


async def call_model(api_url: str, api_key: str, model_name: str, messages: list[dict], model_id: int) -> dict:
    breaker = get_breaker(model_id)
    if breaker.state == CircuitState.OPEN:
        raise CircuitOpenError(f"模型 {model_name} 熔断器处于开启状态，暂时不可用")

    @retry(
        stop=stop_after_attempt(settings.MODEL_CALL_MAX_RETRIES),
        wait=wait_exponential(
            min=settings.MODEL_CALL_RETRY_WAIT_MIN,
            max=settings.MODEL_CALL_RETRY_WAIT_MAX,
        ),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _do_call() -> dict:
        async with httpx.AsyncClient(timeout=settings.MODEL_CALL_TIMEOUT) as client:
            resp = await client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model_name, "messages": messages},
            )
            resp.raise_for_status()
            return resp.json()

    try:
        result = await _do_call()
        breaker.record_success()
        return result
    except Exception:
        breaker.record_failure()
        raise
