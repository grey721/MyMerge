"""Backend-neutral Adapter registry and source-resource lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..core.config import AdapterConfig
from ..core.policies import FusionPolicy, active_adapter_names
from .algebra import SourceAdapterDescriptor
from .resources import SourceAdapterResource


class AdapterPoolState(str, Enum):
    BUILDING = "building"
    SEALED = "sealed"
    PRUNED = "pruned"
    CLOSED = "closed"


class _LogicalSourceAdapter(SourceAdapterResource):
    """Zero-allocation resource used by Mock and metadata-only pools."""

    def __init__(self, coordinate_index: int, spec: AdapterConfig) -> None:
        self._coordinate_index = int(coordinate_index)
        self._spec = spec
        self._closed = False

    @property
    def adapter_key(self) -> str:
        return self._spec.name

    @property
    def coordinate_index(self) -> int:
        return self._coordinate_index

    @property
    def spec(self) -> AdapterConfig:
        return self._spec

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def resident_bytes(self) -> int:
        return 0

    @property
    def device(self) -> Optional[str]:
        return None

    def close(self) -> None:
        self._closed = True


@dataclass(frozen=True)
class AdapterPoolReleaseReport:
    """Result of one explicit or reference-counted source release."""

    supported: bool
    released_adapter_keys: Tuple[str, ...]
    released_coordinate_indices: Tuple[int, ...]
    released_source_bytes: int
    resident_adapter_keys: Tuple[str, ...]
    resident_coordinate_indices: Tuple[int, ...]
    resident_source_bytes: int

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "supported": self.supported,
            "released_adapter_keys": list(self.released_adapter_keys),
            "released_coordinate_indices": list(
                self.released_coordinate_indices
            ),
            "released_source_count": len(self.released_adapter_keys),
            "released_source_bytes": self.released_source_bytes,
            "resident_adapter_keys": list(self.resident_adapter_keys),
            "resident_coordinate_indices": list(
                self.resident_coordinate_indices
            ),
            "resident_source_count": len(self.resident_adapter_keys),
            "resident_source_bytes": self.resident_source_bytes,
        }


class AdapterPool:
    """Stable key registry with an explicitly frozen coordinate layout.

    Adapter names are stable ``adapter_key`` values. A zero-based
    ``coordinate_index`` is assigned only for dense chromosome/runtime arrays.
    Releasing a resource never reindexes the configured coordinate layout.
    """

    resource_kind = "logical"

    def __init__(
        self,
        adapters: Sequence[AdapterConfig] = (),
        *,
        auto_seal: bool = True,
    ) -> None:
        self._specs = []  # type: list[AdapterConfig]
        self._keys = []  # type: list[str]
        self._spec_by_key = {}  # type: Dict[str, AdapterConfig]
        self._coordinate_by_key = {}  # type: Dict[str, int]
        self._resources = {}  # type: Dict[str, SourceAdapterResource]
        self._state = AdapterPoolState.BUILDING
        self._close_count = 0
        self._unreferenced_release_count = 0
        self._released_source_count = 0
        self._released_source_bytes = 0
        self._last_reference_counts = {}  # type: Dict[str, int]
        for adapter in adapters:
            self.add_source(adapter)
        if auto_seal:
            self.seal()

    @property
    def state(self) -> AdapterPoolState:
        return self._state

    @property
    def specs(self) -> Tuple[AdapterConfig, ...]:
        return tuple(self._specs)

    @property
    def adapter_keys(self) -> Tuple[str, ...]:
        return tuple(self._keys)

    @property
    def adapter_names(self) -> Tuple[str, ...]:
        """Policy-facing Adapter name axis; every name is a stable key."""

        return self.adapter_keys

    @property
    def size(self) -> int:
        return len(self._specs)

    @property
    def closed(self) -> bool:
        return self._state == AdapterPoolState.CLOSED

    @property
    def resident_adapter_keys(self) -> Tuple[str, ...]:
        return tuple(
            key
            for key in self._keys
            if key in self._resources and not self._resources[key].closed
        )

    @property
    def resident_coordinate_indices(self) -> Tuple[int, ...]:
        return tuple(
            self.coordinate_index(key) for key in self.resident_adapter_keys
        )

    @property
    def resident_adapter_names(self) -> Tuple[str, ...]:
        return self.resident_adapter_keys

    def coordinate_index(self, adapter_key: str) -> int:
        try:
            return self._coordinate_by_key[adapter_key]
        except KeyError as exc:
            raise KeyError("Unknown Adapter key {!r}".format(adapter_key)) from exc

    def adapter_key(self, coordinate_index: int) -> str:
        normalized = self._normalize_coordinate_index(coordinate_index)
        return self._keys[normalized]

    def spec(self, adapter_key: str) -> AdapterConfig:
        try:
            return self._spec_by_key[adapter_key]
        except KeyError as exc:
            raise KeyError("Unknown Adapter key {!r}".format(adapter_key)) from exc

    def get(self, adapter_key: str) -> SourceAdapterResource:
        """Return a resident source resource by stable key."""

        self._require_not_closed()
        if adapter_key not in self._spec_by_key:
            raise KeyError("Unknown Adapter key {!r}".format(adapter_key))
        resource = self._resources[adapter_key]
        if resource.closed:
            raise KeyError(
                "Adapter resource {!r} is not resident".format(adapter_key)
            )
        return resource

    def add_source(self, adapter: AdapterConfig) -> SourceAdapterResource:
        """Append a source while BUILDING; coordinates freeze at ``seal``."""

        self._require_state(AdapterPoolState.BUILDING)
        key = str(adapter.name)
        if not key:
            raise ValueError("Adapter key cannot be empty")
        if key in self._spec_by_key:
            raise ValueError("Adapter key {!r} is not unique".format(key))
        coordinate_index = len(self._specs)
        resource = self._create_source_resource(coordinate_index, adapter)
        if resource.adapter_key != key:
            raise ValueError("Source resource key does not match Adapter config")
        if resource.coordinate_index != coordinate_index:
            raise ValueError("Source resource coordinate does not match pool")
        self._specs.append(adapter)
        self._keys.append(key)
        self._spec_by_key[key] = adapter
        self._coordinate_by_key[key] = coordinate_index
        self._resources[key] = resource
        return resource

    def seal(self) -> None:
        """Freeze the configured coordinate layout before search starts."""

        if self._state in (AdapterPoolState.SEALED, AdapterPoolState.PRUNED):
            return
        self._require_state(AdapterPoolState.BUILDING)
        if not self._specs:
            raise ValueError("AdapterPool requires at least one Adapter")
        self._state = AdapterPoolState.SEALED

    def validate_policy(self, policy: FusionPolicy) -> None:
        self._require_search_ready()
        actual = tuple(policy.adapter_names)
        if actual != self.adapter_keys:
            raise ValueError(
                "Fusion policy Adapter axis {} does not match configured keys {}"
                .format(actual, self.adapter_keys)
            )
        resident = set(self.resident_adapter_keys)
        unavailable = tuple(
            key for key in active_adapter_names(policy) if key not in resident
        )
        if unavailable:
            raise RuntimeError(
                "Fusion policy activates released source Adapter(s): {}".format(
                    ", ".join(unavailable)
                )
            )

    def release(self, adapter_key: str) -> AdapterPoolReleaseReport:
        """Release one source by stable key."""

        self._require_search_ready()
        if adapter_key not in self._spec_by_key:
            raise KeyError("Unknown Adapter key {!r}".format(adapter_key))
        resource = self._resources[adapter_key]
        if resource.closed:
            return self._release_report(True, (), (), 0)
        self._before_release((adapter_key,))
        resident_bytes = resource.resident_bytes
        resource.close()
        self._released_source_count += 1
        self._released_source_bytes += resident_bytes
        self._state = AdapterPoolState.PRUNED
        return self._release_report(
            True,
            (adapter_key,),
            (resource.coordinate_index,),
            resident_bytes,
        )

    def release_unreferenced(
        self, reference_counts: Mapping[str, int]
    ) -> AdapterPoolReleaseReport:
        """Release every resident source whose stable key has zero references."""

        self._require_search_ready()
        counts = self._validate_reference_counts(reference_counts)
        self._last_reference_counts = dict(counts)
        release_keys = tuple(
            key for key in self.resident_adapter_keys if counts[key] == 0
        )
        self._before_release(release_keys)
        released_keys = []
        released_indices = []
        released_bytes = 0
        failures = []
        for key in release_keys:
            resource = self._resources[key]
            resource_bytes = resource.resident_bytes
            try:
                resource.close()
            except Exception as exc:
                failures.append((key, exc))
            else:
                released_keys.append(key)
                released_indices.append(resource.coordinate_index)
                released_bytes += resource_bytes
        self._unreferenced_release_count += 1
        self._released_source_count += len(released_keys)
        self._released_source_bytes += released_bytes
        if released_keys:
            self._state = AdapterPoolState.PRUNED
        if failures:
            raise RuntimeError(
                "Failed to release unreferenced Adapter resource(s): {}".format(
                    "; ".join(
                        "{}: {}".format(key, exc) for key, exc in failures
                    )
                )
            )
        return self._release_report(
            True,
            tuple(released_keys),
            tuple(released_indices),
            released_bytes,
        )

    def runtime_snapshot(self) -> Mapping[str, Any]:
        resident_keys = self.resident_adapter_keys if not self.closed else ()
        return {
            "implementation": type(self).__name__,
            "resource_kind": self.resource_kind,
            "state": self._state.value,
            "configured_source_count": self.size,
            "resident_source_count": len(resident_keys),
            "resident_source_bytes": self._resident_source_bytes(),
            "adapter_keys": list(self.adapter_keys),
            "coordinate_index_by_key": dict(self._coordinate_by_key),
            "resident_adapter_keys": list(resident_keys),
            "resident_coordinate_indices": list(
                self.resident_coordinate_indices if not self.closed else ()
            ),
            "released_adapter_keys": [
                key for key in self._keys if self._resources[key].closed
            ],
            "device": self._resident_device(),
            "unreferenced_release_count": self._unreferenced_release_count,
            "released_source_count": self._released_source_count,
            "released_source_bytes": self._released_source_bytes,
            "last_reference_counts": dict(self._last_reference_counts),
            "close_count": self._close_count,
        }

    def close(self) -> None:
        if self.closed:
            return
        failures = []
        for key in self.resident_adapter_keys:
            try:
                self._resources[key].close()
            except Exception as exc:
                failures.append((key, exc))
        if failures:
            raise RuntimeError(
                "Failed to close Adapter resource(s): {}".format(
                    "; ".join(
                        "{}: {}".format(key, exc) for key, exc in failures
                    )
                )
            )
        self._state = AdapterPoolState.CLOSED
        self._close_count += 1

    def _create_source_resource(
        self, coordinate_index: int, adapter: AdapterConfig
    ) -> SourceAdapterResource:
        return _LogicalSourceAdapter(coordinate_index, adapter)

    def _before_release(self, adapter_keys: Sequence[str]) -> None:
        """Backend hook executed once before a release set is closed."""

    def _resident_source_bytes(self) -> int:
        if self.closed:
            return 0
        return sum(
            self._resources[key].resident_bytes
            for key in self.resident_adapter_keys
        )

    def _resident_device(self) -> Optional[str]:
        if self.closed:
            return None
        devices = tuple(
            dict.fromkeys(
                self._resources[key].device
                for key in self.resident_adapter_keys
                if self._resources[key].device is not None
            )
        )
        if not devices:
            return None
        return devices[0] if len(devices) == 1 else "mixed"

    def _release_report(
        self,
        supported: bool,
        released_keys: Tuple[str, ...],
        released_indices: Tuple[int, ...],
        released_bytes: int,
    ) -> AdapterPoolReleaseReport:
        return AdapterPoolReleaseReport(
            supported=supported,
            released_adapter_keys=released_keys,
            released_coordinate_indices=released_indices,
            released_source_bytes=int(released_bytes),
            resident_adapter_keys=self.resident_adapter_keys,
            resident_coordinate_indices=self.resident_coordinate_indices,
            resident_source_bytes=self._resident_source_bytes(),
        )

    def _validate_reference_counts(
        self, reference_counts: Mapping[str, int]
    ) -> Dict[str, int]:
        counts = {}
        for raw_key, raw_count in reference_counts.items():
            key = str(raw_key)
            if key not in self._spec_by_key:
                raise KeyError("Unknown Adapter key {!r}".format(key))
            if key in counts:
                raise ValueError("Duplicate Adapter reference key")
            if isinstance(raw_count, bool) or not isinstance(raw_count, int):
                raise TypeError("Adapter reference counts must be integers")
            if raw_count < 0:
                raise ValueError("Adapter reference counts cannot be negative")
            counts[key] = raw_count
        if set(counts) != set(self._keys):
            raise ValueError(
                "Reference counts must define every configured Adapter key"
            )
        return counts

    def _normalize_coordinate_index(self, coordinate_index: int) -> int:
        if isinstance(coordinate_index, bool) or not isinstance(
            coordinate_index, int
        ):
            raise TypeError("coordinate_index must be an integer")
        if coordinate_index < 0 or coordinate_index >= self.size:
            raise KeyError(
                "Unknown Adapter coordinate {}".format(coordinate_index)
            )
        return coordinate_index

    def _require_state(self, expected: AdapterPoolState) -> None:
        if self._state != expected:
            raise RuntimeError(
                "AdapterPool state is {}, expected {}".format(
                    self._state.value, expected.value
                )
            )

    def _require_search_ready(self) -> None:
        if self.closed:
            raise RuntimeError("{} is closed".format(type(self).__name__))
        if self._state not in (
            AdapterPoolState.SEALED,
            AdapterPoolState.PRUNED,
        ):
            raise RuntimeError(
                "AdapterPool must be sealed before search-time access"
            )

    def _require_not_closed(self) -> None:
        if self.closed:
            raise RuntimeError("{} is closed".format(type(self).__name__))

    def __enter__(self) -> "AdapterPool":
        self._require_not_closed()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class ResidentAdapterPool(AdapterPool):
    """Pool whose entries own backend allocations instead of metadata only."""

    resource_kind = "resident"

    @property
    def descriptors(self) -> Tuple[SourceAdapterDescriptor, ...]:
        self._require_search_ready()
        descriptors = []
        for key in self.resident_adapter_keys:
            descriptor = self._resources[key].descriptor
            if descriptor is None:
                raise RuntimeError(
                    "Resident Adapter {!r} has no algebra descriptor".format(key)
                )
            descriptors.append(descriptor)
        return tuple(descriptors)

    @property
    def resident_source_count(self) -> int:
        return 0 if self.closed else len(self.resident_adapter_keys)

    @property
    def resident_source_bytes(self) -> int:
        return self._resident_source_bytes()

    @property
    def device(self) -> Optional[str]:
        return self._resident_device()

    def descriptor(self, adapter_key: str) -> SourceAdapterDescriptor:
        resource = self.get(adapter_key)
        if resource.descriptor is None:
            raise RuntimeError(
                "Resident Adapter {!r} has no algebra descriptor".format(
                    adapter_key
                )
            )
        return resource.descriptor

    @property
    def total_rank(self) -> int:
        return sum(item.rank for item in self.descriptors)

    def runtime_snapshot(self) -> Mapping[str, Any]:
        snapshot = dict(super().runtime_snapshot())
        snapshot["total_source_rank"] = (
            sum(item.rank for item in self.descriptors)
            if not self.closed
            else 0
        )
        return snapshot


__all__ = [
    "AdapterPool",
    "AdapterPoolReleaseReport",
    "AdapterPoolState",
    "ResidentAdapterPool",
]
