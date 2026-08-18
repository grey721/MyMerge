"""Shared lifecycle contracts for source and fused Adapter resources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from ..core.config import AdapterConfig
from ..core.policies import FusionPolicy
from .algebra import SourceAdapterDescriptor


class AdapterResource(ABC):
    """One backend-owned, independently closable Adapter allocation."""

    @property
    @abstractmethod
    def adapter_key(self) -> str:
        """Stable human-readable identity, independent of pool order."""

    @property
    @abstractmethod
    def closed(self) -> bool:
        pass

    @property
    @abstractmethod
    def resident_bytes(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> Optional[str]:
        pass

    @abstractmethod
    def close(self) -> None:
        """Release this resource idempotently."""

    def __enter__(self) -> "AdapterResource":
        if self.closed:
            raise RuntimeError(
                "Adapter resource {!r} is closed".format(self.adapter_key)
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class SourceAdapterResource(AdapterResource):
    """Long-lived source Adapter occupying one global coordinate."""

    @property
    @abstractmethod
    def coordinate_index(self) -> int:
        pass

    @property
    @abstractmethod
    def spec(self) -> AdapterConfig:
        pass

    @property
    def descriptor(self) -> Optional[SourceAdapterDescriptor]:
        return None


class FusedAdapterResource(AdapterResource):
    """Candidate-scoped Adapter produced from one immutable policy."""

    @property
    @abstractmethod
    def source_keys(self) -> Tuple[str, ...]:
        pass

    @property
    @abstractmethod
    def policy_fingerprint(self) -> str:
        pass


class FusionAdapterFactory(ABC):
    """Backend executor for a configured fusion algebra."""

    @abstractmethod
    def create(self, policy: FusionPolicy) -> FusedAdapterResource:
        """Materialize and return one independently closable fused Adapter."""


__all__ = [
    "AdapterResource",
    "FusedAdapterResource",
    "FusionAdapterFactory",
    "SourceAdapterResource",
]
