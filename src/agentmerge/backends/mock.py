"""Fast deterministic backend for topology tests and integration smoke runs."""

from __future__ import annotations

import hashlib
import re
import time
from typing import Sequence, Tuple

from ..core.config import AdapterConfig
from ..core.domain import GenerationRequest, GenerationResult
from ..core.policies import FusionPolicy
from .base import ModelBackend
from .pool import AdapterPool


class MockModelBackend(ModelBackend):
    """Deterministic runtime that never loads a language model.

    It is deliberately simple: if the prompt contains ``[expected: ...]`` it
    returns that value; otherwise it emits a stable diagnostic response. The
    synthetic benchmark bypasses text generation and is the preferred optimizer
    smoke test.
    """

    _EXPECTED_PATTERN = re.compile(r"\[expected:\s*(.*?)\]", re.IGNORECASE)

    def __init__(self, adapters: Sequence[AdapterConfig]) -> None:
        self._pool = AdapterPool(adapters)

    @property
    def adapter_names(self) -> Tuple[str, ...]:
        return self._pool.adapter_names

    @property
    def adapter_specs(self) -> Tuple[AdapterConfig, ...]:
        return self._pool.specs

    def generate(
        self, request: GenerationRequest, mixture: FusionPolicy
    ) -> GenerationResult:
        self._pool.validate_policy(mixture)
        start = time.perf_counter()
        expected = self._EXPECTED_PATTERN.search(request.user_prompt)
        if expected:
            text = expected.group(1).strip()
        else:
            digest = hashlib.sha256(
                (
                    request.agent_name
                    + request.user_prompt
                    + repr(mixture.as_dict())
                ).encode("utf-8")
            ).hexdigest()[:12]
            text = "{}:{}".format(request.agent_name, digest)
        latency = time.perf_counter() - start
        prompt_tokens = _rough_token_count(
            request.system_prompt + "\n" + request.user_prompt
        )
        completion_tokens = _rough_token_count(text)
        return GenerationResult(
            text=text,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={"backend": "mock", "mixture": mixture.as_dict()},
        )

    def runtime_snapshot(self):
        snapshot = dict(super().runtime_snapshot())
        snapshot["adapter_pool"] = dict(self._pool.runtime_snapshot())
        return snapshot

    def close(self) -> None:
        self._pool.close()


def _rough_token_count(text: str) -> int:
    return max(1, len(text.split()))
