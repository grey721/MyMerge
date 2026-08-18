"""Typed experiment configuration and validation."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is internally inconsistent."""


def _required(data: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in data:
        raise ConfigError("Missing required key {!r} in {}".format(key, context))
    return data[key]


def _reject_unknown_keys(
    data: Mapping[str, Any], allowed: Sequence[str], context: str
) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(
            "Unknown key(s) in {}: {}".format(context, ", ".join(unknown))
        )


def _expect_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError("{} must be a mapping".format(context))
    return value


def _expect_mapping_list(
    value: Any, context: str
) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError("{} must be a list".format(context))
    result = []
    for index, item in enumerate(value):
        result.append(_expect_mapping(item, "{}[{}]".format(context, index)))
    return tuple(result)


def _validate_config_schema(raw: Mapping[str, Any]) -> None:
    """Reject misspelled or unsupported fields in the current schema."""

    _reject_unknown_keys(
        raw,
        (
            "name",
            "seed",
            "output_dir",
            "model",
            "adapters",
            "agents",
            "topology",
            "benchmark",
            "objectives",
            "runtime",
            "metadata",
        ),
        "experiment",
    )
    if "metadata" in raw:
        _expect_mapping(raw["metadata"], "experiment.metadata")

    if "model" in raw:
        model = _expect_mapping(raw["model"], "model")
        _reject_unknown_keys(
            model,
            (
                "backend",
                "name_or_path",
                "base_model_id",
                "dtype",
                "device_map",
                "trust_remote_code",
                "generation",
                "options",
            ),
            "model",
        )
        if "generation" in model:
            generation = _expect_mapping(model["generation"], "model.generation")
            _reject_unknown_keys(
                generation,
                ("max_new_tokens", "temperature", "enable_thinking", "seed"),
                "model.generation",
            )
        if "options" in model:
            _expect_mapping(model["options"], "model.options")

    item_schemas = (
        (
            "adapters",
            ("name", "path", "base_model", "rank", "target_modules", "metadata"),
        ),
        (
            "agents",
            (
                "name",
                "role",
                "system_prompt",
                "prompt_template",
                "max_new_tokens",
                "metadata",
            ),
        ),
        ("objectives", ("metric", "direction")),
    )
    for section, allowed in item_schemas:
        if section not in raw:
            continue
        for index, item in enumerate(_expect_mapping_list(raw[section], section)):
            context = "{}[{}]".format(section, index)
            _reject_unknown_keys(item, allowed, context)
            if "metadata" in item:
                _expect_mapping(item["metadata"], context + ".metadata")

    if "topology" in raw:
        topology = _expect_mapping(raw["topology"], "topology")
        _reject_unknown_keys(
            topology,
            (
                "kind",
                "edges",
                "output_agent",
                "loop_agents",
                "max_steps",
                "max_context_tokens",
            ),
            "topology",
        )

    if "benchmark" in raw:
        benchmark = _expect_mapping(raw["benchmark"], "benchmark")
        _reject_unknown_keys(
            benchmark,
            (
                "kind",
                "path",
                "input_field",
                "target_field",
                "group_field",
                "id_field",
                "max_cases",
                "options",
            ),
            "benchmark",
        )
        if "options" in benchmark:
            _expect_mapping(benchmark["options"], "benchmark.options")

    if "runtime" in raw:
        runtime = _expect_mapping(raw["runtime"], "runtime")
        _reject_unknown_keys(
            runtime,
            (
                "mode",
                "gpu_ids",
                "start_method",
                "preload_workers",
                "evaluation_timeout_seconds",
                "max_retries",
                "cpu_threads_per_worker",
                "worker_env",
            ),
            "runtime",
        )
        if "worker_env" in runtime:
            _expect_mapping(runtime["worker_env"], "runtime.worker_env")

@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    enable_thinking: bool = False
    seed: Optional[int] = None


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    name_or_path: str
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    options: Mapping[str, Any] = field(default_factory=dict)
    # Stable model identity used by PEFT adapter compatibility checks.  This
    # may remain ``Qwen/Qwen3-8B`` even when ``name_or_path`` is a shared local
    # snapshot outside the project directory.
    base_model_id: Optional[str] = None

    @property
    def base_model_identity(self) -> str:
        return self.base_model_id or self.name_or_path

    @property
    def base_model_aliases(self) -> Tuple[str, ...]:
        """Accepted identities for adapters built from this model snapshot."""

        values = [self.base_model_identity, self.name_or_path]
        return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class AdapterConfig:
    name: str
    path: str
    base_model: Optional[str] = None
    rank: Optional[int] = None
    target_modules: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    name: str
    role: str
    system_prompt: str
    prompt_template: str = (
        "Task:\n{task}\n\nMessages from upstream agents:\n{incoming}"
    )
    max_new_tokens: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyConfig:
    edges: Tuple[Tuple[str, str], ...]
    output_agent: str
    kind: str = "tool_loop"
    loop_agents: Tuple[str, ...] = ()
    max_steps: int = 15
    max_context_tokens: int = 4096


@dataclass(frozen=True)
class BenchmarkConfig:
    kind: str
    path: Optional[str] = None
    input_field: str = "input"
    target_field: str = "target"
    group_field: str = "group"
    id_field: str = "id"
    max_cases: Optional[int] = None
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def inference_batch_size(self) -> int:
        """Single source of truth for evaluator and backend batch capacity."""

        return int(self.options.get("inference_batch_size", 8))


@dataclass(frozen=True)
class ObjectiveConfig:
    metric: str
    direction: str

    def __post_init__(self) -> None:
        metric = str(self.metric).strip()
        direction = str(self.direction).strip().lower()
        if not metric:
            raise ConfigError("Objective metric cannot be empty")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "direction", direction)
        if self.direction not in ("maximize", "minimize"):
            raise ConfigError(
                "Objective direction must be 'maximize' or 'minimize', got {!r}".format(
                    self.direction
                )
            )


@dataclass(frozen=True)
class SearchConfig:
    """CLI-resolved search request, never authored in experiment YAML.

    ``granularities`` describes one or more sequential stages.  The optimizer
    package owns the chromosome, representation and stage-transition policy.
    ``options.<algorithm>`` contains only direct CLI algorithm parameters.
    """

    algorithm: str = "nsga2"
    population_size: int = 24
    generations: int = 10
    workers: int = 1
    seed: int = 42
    checkpoint_every: int = 1
    granularities: Tuple[str, ...] = ("global",)
    stage_population_sizes: Tuple[int, ...] = ()
    stage_generations: Tuple[int, ...] = ()
    granularity_options: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def stage_count(self) -> int:
        return len(self.granularities)

    def population_size_for(self, stage_index: int) -> int:
        if self.stage_population_sizes:
            return self.stage_population_sizes[stage_index]
        return self.population_size

    def generations_for(self, stage_index: int) -> int:
        if self.stage_generations:
            return self.stage_generations[stage_index]
        return self.generations

    @property
    def total_generation_batches(self) -> int:
        return sum(
            self.generations_for(index)
            for index in range(self.stage_count)
        )

    @property
    def final_generation_index(self) -> int:
        return self.total_generation_batches - 1


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution placement independently of the evolutionary algorithm."""

    mode: str = "local"
    gpu_ids: Tuple[str, ...] = ()
    start_method: str = "spawn"
    preload_workers: bool = True
    evaluation_timeout_seconds: Optional[float] = None
    max_retries: int = 0
    cpu_threads_per_worker: int = 4
    worker_env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: str
    model: ModelConfig
    adapters: Tuple[AdapterConfig, ...]
    agents: Tuple[AgentConfig, ...]
    topology: TopologyConfig
    benchmark: BenchmarkConfig
    objectives: Tuple[ObjectiveConfig, ...]
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    seed: int = 42
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def adapter_names(self) -> Tuple[str, ...]:
        return tuple(adapter.name for adapter in self.adapters)

    @property
    def agent_names(self) -> Tuple[str, ...]:
        return tuple(agent.name for agent in self.agents)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload

    def to_experiment_dict(self) -> Dict[str, Any]:
        """Serialize only the YAML-owned experiment definition."""

        payload = self.to_dict()
        payload.pop("search", None)
        topology = payload["topology"]
        if self.topology.kind == "fixed":
            topology.pop("loop_agents", None)
            topology.pop("max_steps", None)
            topology.pop("max_context_tokens", None)
        elif self.topology.kind == "tool_loop":
            topology.pop("edges", None)
        return payload

    def validate(self) -> None:
        backend_name = self.model.backend.lower()
        if backend_name not in ("mock", "peft", "vllm"):
            raise ConfigError(
                "model.backend must be mock, peft, or vllm; got {!r}".format(
                    self.model.backend
                )
            )
        if not self.adapters:
            raise ConfigError("At least one adapter is required")
        if not self.agents:
            raise ConfigError("At least one agent is required")
        if not self.objectives:
            raise ConfigError("At least one objective is required")
        if self.benchmark.inference_batch_size < 1:
            raise ConfigError(
                "benchmark.options.inference_batch_size must be positive"
            )

        _assert_unique(self.adapter_names, "adapter")
        _assert_unique(self.agent_names, "agent")
        _assert_unique(
            tuple(objective.metric for objective in self.objectives),
            "objective metric",
        )

        agent_set = set(self.agent_names)
        if self.topology.output_agent not in agent_set:
            raise ConfigError("topology.output_agent must name a configured agent")
        topology_kind = self.topology.kind.lower().replace("-", "_")
        if topology_kind == "fixed":
            if self.topology.loop_agents:
                raise ConfigError(
                    "Fixed topology cannot define topology.loop_agents"
                )
            for source, target in self.topology.edges:
                if source not in agent_set or target not in agent_set:
                    raise ConfigError(
                        "Topology edge ({!r}, {!r}) references an unknown agent".format(
                            source, target
                        )
                    )
                if source == target:
                    raise ConfigError("Topology cannot contain self edges")
            _validate_dag(self.agent_names, self.topology.edges)
        elif topology_kind == "tool_loop":
            if self.topology.edges:
                raise ConfigError(
                    "Tool-loop topology uses loop_agents instead of edges"
                )
            if len(self.topology.loop_agents) != 2:
                raise ConfigError(
                    "Tool-loop topology requires exactly two loop_agents: "
                    "planner and argument generator"
                )
            if len(set(self.topology.loop_agents)) != 2:
                raise ConfigError("topology.loop_agents must be distinct")
            unknown = set(self.topology.loop_agents) - agent_set
            if unknown:
                raise ConfigError(
                    "topology.loop_agents references unknown agents: {}".format(
                        ", ".join(sorted(unknown))
                    )
                )
            used_agents = set(self.topology.loop_agents)
            used_agents.add(self.topology.output_agent)
            if used_agents != agent_set:
                raise ConfigError(
                    "Tool-loop topology must use every configured agent; unused: {}".format(
                        ", ".join(sorted(agent_set - used_agents))
                    )
                )
            if self.topology.max_steps < 1:
                raise ConfigError("topology.max_steps must be positive")
            if self.topology.max_context_tokens < 1:
                raise ConfigError(
                    "topology.max_context_tokens must be positive"
                )
        else:
            raise ConfigError(
                "topology.kind must be fixed or tool_loop; got {!r}".format(
                    self.topology.kind
                )
            )

        if self.search.population_size < 1:
            raise ConfigError("search.population_size must be positive")
        if self.search.generations < 1:
            raise ConfigError("search.generations must be positive")
        if self.search.workers < 1:
            raise ConfigError("search.workers must be positive")
        if not self.search.granularities:
            raise ConfigError("At least one search granularity is required")
        if self.search.stage_population_sizes and len(
            self.search.stage_population_sizes
        ) != self.search.stage_count:
            raise ConfigError(
                "Stage population counts must match search granularities"
            )
        if self.search.stage_generations and len(
            self.search.stage_generations
        ) != self.search.stage_count:
            raise ConfigError(
                "Stage generation counts must match search granularities"
            )
        if any(
            self.search.population_size_for(index) < 1
            for index in range(self.search.stage_count)
        ):
            raise ConfigError("Stage populations must be positive")
        if any(
            self.search.generations_for(index) < 1
            for index in range(self.search.stage_count)
        ):
            raise ConfigError("Stage generations must be positive")
        if self.runtime.mode not in ("local", "multi_gpu"):
            raise ConfigError("runtime.mode must be 'local' or 'multi_gpu'")
        if self.runtime.mode == "multi_gpu" and not self.runtime.gpu_ids:
            raise ConfigError(
                "runtime.gpu_ids cannot be empty in multi_gpu mode"
            )
        _assert_unique(self.runtime.gpu_ids, "GPU")
        if self.runtime.start_method not in ("spawn", "forkserver"):
            raise ConfigError(
                "runtime.start_method must be 'spawn' or 'forkserver'"
            )
        if (
            self.runtime.evaluation_timeout_seconds is not None
            and self.runtime.evaluation_timeout_seconds <= 0
        ):
            raise ConfigError(
                "runtime.evaluation_timeout_seconds must be positive"
            )
        if self.runtime.max_retries < 0:
            raise ConfigError("runtime.max_retries cannot be negative")
        if self.runtime.cpu_threads_per_worker < 1:
            raise ConfigError(
                "runtime.cpu_threads_per_worker must be positive"
            )
        if "CUDA_VISIBLE_DEVICES" in self.runtime.worker_env:
            raise ConfigError(
                "runtime.worker_env cannot override CUDA_VISIBLE_DEVICES"
            )
        if (
            self.runtime.mode == "multi_gpu"
            and backend_name == "peft"
            and self.model.device_map not in ("auto", "cuda", "cuda:0")
        ):
            raise ConfigError(
                "Multi-GPU PEFT workers require device_map auto/cuda/cuda:0; "
                "each process sees its assigned physical GPU as logical cuda:0"
            )
        if backend_name == "vllm":
            vllm_options = self.model.options.get("vllm", {})
            if not isinstance(vllm_options, Mapping):
                raise ConfigError("model.options.vllm must be a mapping")
            if str(vllm_options.get("lora_transport", "memory")).lower() != "memory":
                raise ConfigError(
                    "The current vLLM backend requires lora_transport=memory"
                )
            engine_mode = str(vllm_options.get("engine_mode", "v1")).lower()
            if engine_mode not in ("v0", "v1"):
                raise ConfigError(
                    "model.options.vllm.engine_mode must be v0 or v1"
                )
            if int(vllm_options.get("tensor_parallel_size", 1)) != 1:
                raise ConfigError(
                    "The vLLM memory backend requires tensor_parallel_size=1; "
                    "parallelize with runtime.gpu_ids instead"
                )
            source_device = str(
                vllm_options.get("source_adapter_device", "cuda")
            ).lower()
            if source_device not in ("cpu", "cuda", "cuda:0"):
                raise ConfigError(
                    "model.options.vllm.source_adapter_device must be cpu or cuda"
                )
            max_loras = int(vllm_options.get("max_loras", 1))
            if max_loras < 1:
                raise ConfigError("model.options.vllm.max_loras must be positive")
            if "max_num_seqs" in vllm_options:
                raise ConfigError(
                    "model.options.vllm.max_num_seqs is derived from "
                    "benchmark.options.inference_batch_size and cannot be "
                    "configured independently"
                )
            kv_cache_memory_bytes = vllm_options.get("kv_cache_memory_bytes")
            if (
                kv_cache_memory_bytes is not None
                and int(kv_cache_memory_bytes) < 1
            ):
                raise ConfigError(
                    "model.options.vllm.kv_cache_memory_bytes must be positive"
                )
            max_cpu_loras = int(
                vllm_options.get(
                    "max_cpu_loras",
                    len(self.adapters) + max(4, len(self.agents)),
                )
            )
            required_cpu_loras = len(self.adapters) + len(self.agents)
            if max_cpu_loras < required_cpu_loras:
                raise ConfigError(
                    "model.options.vllm.max_cpu_loras must be at least {} "
                    "(source pool {} + at most one candidate LoRA per Agent "
                    "{})".format(
                        required_cpu_loras,
                        len(self.adapters),
                        len(self.agents),
                    )
                )

        for adapter in self.adapters:
            if (
                adapter.base_model
                and adapter.base_model not in self.model.base_model_aliases
                and not bool(self.model.options.get("allow_base_model_mismatch", False))
            ):
                raise ConfigError(
                    "Adapter {!r} declares base model {!r}, expected {!r}".format(
                        adapter.name,
                        adapter.base_model,
                        self.model.base_model_identity,
                    )
                )


def _assert_unique(values: Sequence[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise ConfigError("{} names must be unique".format(label.capitalize()))


def _validate_dag(
    agent_names: Sequence[str], edges: Sequence[Tuple[str, str]]
) -> None:
    indegree = {name: 0 for name in agent_names}
    children = {name: [] for name in agent_names}
    for source, target in edges:
        children[source].append(target)
        indegree[target] += 1
    queue = [name for name in agent_names if indegree[name] == 0]
    visited = 0
    while queue:
        current = queue.pop(0)
        visited += 1
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(agent_names):
        raise ConfigError("Topology must be a directed acyclic graph")


def load_experiment_config(path: str) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            raw = json.load(handle)
        else:
            raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ConfigError("Top-level configuration must be a mapping")
    config = experiment_config_from_dict(raw, base_dir=config_path.parent)
    config.validate()
    return config


def apply_experiment_overrides(
    config: ExperimentConfig,
    *,
    run_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed: Optional[int] = None,
    objectives: Optional[Sequence[str]] = None,
    algorithm: Optional[str] = None,
    search_granularity: Optional[Sequence[str]] = None,
    populations: Optional[Sequence[int]] = None,
    generations: Optional[Sequence[int]] = None,
    workers: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
    max_cases: Optional[int] = None,
    batch_size: Optional[int] = None,
    runtime_mode: Optional[str] = None,
    device: Optional[Sequence[str]] = None,
    evaluation_timeout_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
    crossover_probability: Optional[float] = None,
    mutation_probability: Optional[float] = None,
    sparsity_mutation_probability: Optional[float] = None,
    mutation_sigma: Optional[float] = None,
    differential_weight: Optional[float] = None,
    neighborhood_size: Optional[int] = None,
    neighborhood_mating_probability: Optional[float] = None,
    max_replacements: Optional[int] = None,
    decomposition: Optional[str] = None,
    pbi_theta: Optional[float] = None,
    mutation_distribution_index: Optional[float] = None,
    refinement_groups: Optional[Sequence[int]] = None,
    total_budget: Optional[int] = None,
    min_stage_fe: Optional[Sequence[int]] = None,
    first_split: Optional[int] = None,
    num_sectors: Optional[int] = None,
    sector_seed: Optional[int] = None,
    direction_window: Optional[int] = None,
    min_directions_per_sector: Optional[int] = None,
    min_valid_sector_fraction: Optional[float] = None,
    direction_norm_eps: Optional[float] = None,
    coherence_threshold: Optional[float] = None,
    utility_ratio_threshold: Optional[float] = None,
    saturation_patience: Optional[int] = None,
    utility_ema_beta: Optional[float] = None,
    utility_eps: Optional[float] = None,
    checkpoint_interval_fe: Optional[int] = None,
    enable_zero_progress_fallback: Optional[bool] = None,
    utility_zero_eps: Optional[float] = None,
    symmetry_diagnostic_batches: Optional[int] = None,
    min_active_adapters: Optional[int] = None,
    max_active_adapters: Optional[int] = None,
    num_layers: Optional[int] = None,
    group_size: Optional[int] = None,
    group_count: Optional[int] = None,
) -> ExperimentConfig:
    """Resolve model-independent search/runtime controls from the CLI.

    Experiment YAML owns the model, Adapter pool, Agent system and benchmark.
    The optimizer, objectives, fusion granularity, budgets and operators can be
    selected from the CLI while the YAML values remain reproducible defaults.
    """

    search = config.search
    runtime = config.runtime
    benchmark = config.benchmark
    model = config.model
    resolved_objectives = config.objectives

    if objectives is not None:
        if not objectives:
            raise ConfigError("--objective requires at least one value")
        resolved_objectives = tuple(
            _parse_objective_override(value) for value in objectives
        )

    if seed is not None:
        search = replace(search, seed=int(seed))
        model = replace(
            model,
            generation=replace(model.generation, seed=int(seed)),
        )
    if algorithm is not None:
        search = replace(
            search,
            algorithm=str(algorithm).lower().replace("-", "_"),
        )
    implicit_two_stage = (
        search.algorithm in (
            "two_stage_nsga2",
            "two_stage_de",
        )
        and search_granularity is None
    )
    implicit_progressive = (
        search.algorithm == "progressive_de" and search_granularity is None
    )
    implicit_adaptive_resolution = (
        search.algorithm == "adaptive_resolution_de"
        and search_granularity is None
    )
    resolved_refinement_groups = (
        tuple(int(value) for value in refinement_groups)
        if refinement_groups is not None
        else tuple(
            int(value)
            for value in search.options.get("progressive_de", {}).get(
                "refinement_groups", ()
            )
        )
    )
    if implicit_progressive and not resolved_refinement_groups:
        raise ConfigError("progressive_de requires --refinement_groups")
    granularities = (
        tuple(
            str(item).lower().replace("-", "_")
            for item in search_granularity
        )
        if search_granularity is not None
        else (
            (
                ("adapter",)
                + ("layer_partition",)
                * (len(resolved_refinement_groups) - 1)
            )
            if implicit_progressive
            else ("adapter",)
            if implicit_adaptive_resolution
            else
            (
                ("adapter", "layer")
                if search.algorithm == "two_stage_de"
                else ("global", "layer")
            )
            if implicit_two_stage
            else search.granularities
        )
    )
    if populations is not None:
        values = _stage_cli_values(populations)
        stage_values = _broadcast_stage_values(values, len(granularities), "populations")
        search = replace(
            search,
            population_size=stage_values[0],
            stage_population_sizes=stage_values,
        )
    elif (
        search_granularity is not None
        or implicit_two_stage
        or implicit_progressive
        or implicit_adaptive_resolution
    ):
        search = replace(
            search,
            stage_population_sizes=(search.population_size,) * len(granularities),
        )
    if generations is not None:
        values = _stage_cli_values(generations)
        stage_values = _broadcast_stage_values(values, len(granularities), "generations")
        search = replace(
            search,
            generations=stage_values[0],
            stage_generations=stage_values,
        )
    elif (
        search_granularity is not None
        or implicit_two_stage
        or implicit_progressive
        or implicit_adaptive_resolution
    ):
        search = replace(
            search,
            stage_generations=(search.generations,) * len(granularities),
        )
    if (
        search_granularity is not None
        or implicit_two_stage
        or implicit_progressive
        or implicit_adaptive_resolution
    ):
        search = replace(search, granularities=granularities)
    if workers is not None:
        search = replace(search, workers=int(workers))
    if checkpoint_every is not None:
        search = replace(search, checkpoint_every=int(checkpoint_every))
    optimizer_overrides = {
        key: value
        for key, value in {
            "crossover_probability": crossover_probability,
            "mutation_probability": mutation_probability,
            "sparsity_mutation_probability": sparsity_mutation_probability,
            "mutation_sigma": mutation_sigma,
            "differential_weight": differential_weight,
            "neighborhood_size": neighborhood_size,
            "neighborhood_mating_probability": neighborhood_mating_probability,
            "max_replacements": max_replacements,
            "decomposition": decomposition,
            "pbi_theta": pbi_theta,
            "mutation_distribution_index": mutation_distribution_index,
            "min_active_adapters": min_active_adapters,
            "max_active_adapters": max_active_adapters,
            "refinement_groups": (
                resolved_refinement_groups
                if refinement_groups is not None
                else None
            ),
            "total_budget": total_budget,
            "min_stage_fe": (
                tuple(int(value) for value in min_stage_fe)
                if min_stage_fe is not None
                else None
            ),
            "first_split": first_split,
            "num_sectors": num_sectors,
            "sector_seed": sector_seed,
            "direction_window": direction_window,
            "min_directions_per_sector": min_directions_per_sector,
            "min_valid_sector_fraction": min_valid_sector_fraction,
            "direction_norm_eps": direction_norm_eps,
            "coherence_threshold": coherence_threshold,
            "utility_ratio_threshold": utility_ratio_threshold,
            "saturation_patience": saturation_patience,
            "utility_ema_beta": utility_ema_beta,
            "utility_eps": utility_eps,
            "checkpoint_interval_fe": checkpoint_interval_fe,
            "enable_zero_progress_fallback": enable_zero_progress_fallback,
            "utility_zero_eps": utility_zero_eps,
            "symmetry_diagnostic_batches": symmetry_diagnostic_batches,
        }.items()
        if value is not None
    }
    if optimizer_overrides:
        if search.algorithm not in (
            "nsga2",
            "two_stage_nsga2",
            "de",
            "two_stage_de",
            "progressive_de",
            "adaptive_resolution_de",
        ):
            raise ConfigError(
                "Direct optimizer CLI parameters are not supported for "
                "algorithm {!r}".format(search.algorithm)
            )
        options = dict(search.options)
        namespace = dict(options.get(search.algorithm, {}))
        namespace.update(optimizer_overrides)
        options[search.algorithm] = namespace
        search = replace(search, options=options)

    granularity_overrides = {
        key: value
        for key, value in {
            "num_layers": num_layers,
            "group_size": group_size,
            "group_count": group_count,
        }.items()
        if value is not None
    }
    if granularity_overrides:
        search = replace(
            search,
            granularity_options={
                **search.granularity_options,
                **granularity_overrides,
            },
        )

    if max_cases is not None:
        benchmark = replace(benchmark, max_cases=int(max_cases))
    if batch_size is not None:
        benchmark_options = dict(benchmark.options)
        benchmark_options["inference_batch_size"] = int(batch_size)
        benchmark = replace(benchmark, options=benchmark_options)

    if runtime_mode is not None:
        normalized_mode = str(runtime_mode).lower().replace("-", "_")
        runtime = replace(
            runtime,
            mode=normalized_mode,
            gpu_ids=(
                ()
                if normalized_mode == "local" and device is None
                else runtime.gpu_ids
            ),
        )
    if device is not None:
        resolved_devices = resolve_device_ids(device)
        runtime = replace(
            runtime,
            mode="multi_gpu",
            gpu_ids=resolved_devices,
        )
    if evaluation_timeout_seconds is not None:
        runtime = replace(
            runtime,
            evaluation_timeout_seconds=float(evaluation_timeout_seconds),
        )
    if max_retries is not None:
        runtime = replace(runtime, max_retries=int(max_retries))

    resolved_output = config.output_dir
    if output_dir is not None:
        resolved_output = str(Path(output_dir).expanduser().resolve())
    resolved = replace(
        config,
        name=str(run_name) if run_name is not None else config.name,
        output_dir=resolved_output,
        seed=int(seed) if seed is not None else config.seed,
        model=model,
        benchmark=benchmark,
        objectives=resolved_objectives,
        runtime=runtime,
        search=search,
    )
    resolved.validate()
    return resolved


def _parse_objective_override(value: str) -> ObjectiveConfig:
    metric, separator, raw_direction = str(value).rpartition(":")
    if not separator or not metric.strip() or not raw_direction.strip():
        raise ConfigError(
            "--objective must use METRIC:DIRECTION, got {!r}".format(value)
        )
    direction = raw_direction.strip().lower()
    direction = {"max": "maximize", "min": "minimize"}.get(
        direction, direction
    )
    return ObjectiveConfig(metric.strip(), direction)


def _broadcast_stage_values(
    values: Sequence[int], stage_count: int, name: str
) -> Tuple[int, ...]:
    if not values:
        raise ConfigError("--{} requires at least one value".format(name))
    if len(values) == 1:
        return (int(values[0]),) * stage_count
    if len(values) != stage_count:
        raise ConfigError(
            "--{} accepts one shared value or exactly {} stage values".format(
                name, stage_count
            )
        )
    return tuple(int(value) for value in values)


def _stage_cli_values(values: Any) -> Tuple[int, ...]:
    if isinstance(values, (int, float, str)):
        return (int(values),)
    return tuple(int(value) for value in values)


def resolve_device_ids(device: Sequence[str]) -> Tuple[str, ...]:
    """Resolve CLI device IDs, including the explicit ``auto`` selector."""

    values = tuple(str(item) for item in device)
    if len(values) != 1 or values[0].lower() != "auto":
        if any(item.lower() == "auto" for item in values):
            raise ConfigError("--device auto cannot be combined with explicit IDs")
        if not values:
            raise ConfigError("--device requires at least one device ID")
        return values

    discovered = _discover_cuda_device_ids()
    if not discovered:
        raise ConfigError(
            "--device auto could not find a visible CUDA device; "
            "pass explicit physical GPU IDs instead"
        )
    return discovered


def _discover_cuda_device_ids() -> Tuple[str, ...]:
    """Discover physical IDs without importing torch in the CLI process."""

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        try:
            output = subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=index",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
            ids = tuple(
                line.strip()
                for line in output.splitlines()
                if line.strip().isdigit()
            )
            if ids:
                return ids
        except (OSError, subprocess.SubprocessError):
            pass

    try:
        import torch  # type: ignore

        count = int(torch.cuda.device_count())
    except (ImportError, RuntimeError, AttributeError):
        count = 0
    return tuple(str(index) for index in range(count))


def experiment_config_from_dict(
    raw: Mapping[str, Any], base_dir: Optional[Path] = None
) -> ExperimentConfig:
    _validate_config_schema(raw)
    model_raw = _required(raw, "model", "experiment")
    generation_raw = model_raw.get("generation", {})
    generation = GenerationConfig(
        max_new_tokens=int(generation_raw.get("max_new_tokens", 256)),
        temperature=float(generation_raw.get("temperature", 0.0)),
        enable_thinking=bool(generation_raw.get("enable_thinking", False)),
        seed=_optional_int(generation_raw.get("seed")),
    )
    model = ModelConfig(
        backend=str(model_raw.get("backend", "peft")),
        name_or_path=_resolve_path_or_id(
            str(_required(model_raw, "name_or_path", "model")), base_dir
        ),
        base_model_id=_optional_str(model_raw.get("base_model_id")),
        dtype=str(model_raw.get("dtype", "bfloat16")),
        device_map=str(model_raw.get("device_map", "auto")),
        trust_remote_code=bool(model_raw.get("trust_remote_code", False)),
        generation=generation,
        options=dict(model_raw.get("options", {})),
    )

    adapters = tuple(
        AdapterConfig(
            name=str(_required(item, "name", "adapter")),
            path=_resolve_path_or_id(
                str(_required(item, "path", "adapter")), base_dir
            ),
            base_model=_optional_str(item.get("base_model")),
            rank=_optional_int(item.get("rank")),
            target_modules=tuple(str(value) for value in item.get("target_modules", [])),
            metadata=dict(item.get("metadata", {})),
        )
        for item in _required(raw, "adapters", "experiment")
    )

    agents = tuple(
        AgentConfig(
            name=str(_required(item, "name", "agent")),
            role=str(item.get("role", item["name"])),
            system_prompt=str(item.get("system_prompt", "")),
            prompt_template=str(
                item.get(
                    "prompt_template",
                    "Task:\n{task}\n\nMessages from upstream agents:\n{incoming}",
                )
            ),
            max_new_tokens=_optional_int(item.get("max_new_tokens")),
            metadata=dict(item.get("metadata", {})),
        )
        for item in _required(raw, "agents", "experiment")
    )

    topology_raw = _required(raw, "topology", "experiment")
    topology = TopologyConfig(
        edges=tuple(
            (str(edge[0]), str(edge[1])) for edge in topology_raw.get("edges", [])
        ),
        output_agent=str(
            _required(topology_raw, "output_agent", "topology")
        ),
        kind=str(topology_raw.get("kind", "tool_loop"))
        .lower()
        .replace("-", "_"),
        loop_agents=tuple(
            str(agent) for agent in topology_raw.get("loop_agents", [])
        ),
        max_steps=int(topology_raw.get("max_steps", 15)),
        max_context_tokens=int(
            topology_raw.get("max_context_tokens", 4096)
        ),
    )

    benchmark_raw = _required(raw, "benchmark", "experiment")
    benchmark_path = benchmark_raw.get("path")
    benchmark = BenchmarkConfig(
        kind=str(benchmark_raw.get("kind", "jsonl")),
        path=(
            _resolve_path_or_id(str(benchmark_path), base_dir)
            if benchmark_path is not None
            else None
        ),
        input_field=str(benchmark_raw.get("input_field", "input")),
        target_field=str(benchmark_raw.get("target_field", "target")),
        group_field=str(benchmark_raw.get("group_field", "group")),
        id_field=str(benchmark_raw.get("id_field", "id")),
        max_cases=_optional_int(benchmark_raw.get("max_cases")),
        options=dict(benchmark_raw.get("options", {})),
    )

    objectives = tuple(
        ObjectiveConfig(
            metric=str(_required(item, "metric", "objective")),
            direction=str(item.get("direction", "maximize")).lower(),
        )
        for item in _required(raw, "objectives", "experiment")
    )

    search = SearchConfig(
        seed=int(raw.get("seed", 42)),
    )

    runtime_raw = raw.get("runtime", {})
    runtime = RuntimeConfig(
        mode=str(runtime_raw.get("mode", "local")).lower().replace("-", "_"),
        gpu_ids=tuple(str(value) for value in runtime_raw.get("gpu_ids", [])),
        start_method=str(runtime_raw.get("start_method", "spawn")).lower(),
        preload_workers=bool(runtime_raw.get("preload_workers", True)),
        evaluation_timeout_seconds=(
            None
            if runtime_raw.get("evaluation_timeout_seconds") is None
            else float(runtime_raw["evaluation_timeout_seconds"])
        ),
        max_retries=int(runtime_raw.get("max_retries", 0)),
        cpu_threads_per_worker=int(
            runtime_raw.get("cpu_threads_per_worker", 4)
        ),
        worker_env={
            str(key): str(value)
            for key, value in runtime_raw.get("worker_env", {}).items()
        },
    )

    output_dir = str(raw.get("output_dir", "outputs"))
    if base_dir is not None and not Path(output_dir).is_absolute():
        output_dir = str((base_dir / output_dir).resolve())

    return ExperimentConfig(
        name=str(raw.get("name", "agentmerge-experiment")),
        output_dir=output_dir,
        model=model,
        adapters=adapters,
        agents=agents,
        topology=topology,
        benchmark=benchmark,
        objectives=objectives,
        runtime=runtime,
        search=search,
        seed=int(raw.get("seed", 42)),
        metadata=dict(raw.get("metadata", {})),
    )


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _resolve_path_or_id(value: str, base_dir: Optional[Path]) -> str:
    """Resolve explicit local paths while leaving Hugging Face IDs untouched."""

    if base_dir is None:
        return value
    candidate = Path(value).expanduser()
    looks_local = value.startswith((".", "/", "~")) or candidate.exists()
    if looks_local and not candidate.is_absolute():
        return str((base_dir / candidate).resolve())
    if candidate.is_absolute():
        return str(candidate)
    return value
