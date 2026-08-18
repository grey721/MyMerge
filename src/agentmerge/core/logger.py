"""Human-readable logging for one experiment run."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional


_HIDDEN_EVENTS = {
    "candidate_cache_hit",
    "candidate_submitted",
}
_WARNING_EVENTS = {"candidate_retrying", "resolution_symmetry_warning"}
_ERROR_EVENTS = {"evaluation_batch_failed", "run_failed"}


class ExperimentLogger:
    """Write one run's messages to both ``run.log`` and the console."""

    def __init__(
        self,
        run_dir: Path,
        name: str,
        *,
        console: bool = True,
        filename: str = "run.log",
        generation_width: int = 1,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename
        self.generation_width = max(1, int(generation_width))
        logger_name = "agentmerge.run.{}.{}".format(
            _slug(name), abs(hash(str(self.run_dir.resolve())))
        )
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._close_handlers()

        formatter = logging.Formatter(
            "%(levelname)-8s   %(asctime)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        file_handler = logging.FileHandler(str(self.path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

    def debug(self, message: str, *args: Any) -> None:
        self._logger.debug(message, *args)

    def info(self, message: str, *args: Any) -> None:
        self._logger.info(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        self._logger.warning(message, *args)

    def error(self, message: str, *args: Any) -> None:
        self._logger.error(message, *args)

    def exception(self, message: str, *args: Any) -> None:
        self._logger.exception(message, *args)

    def event(
        self,
        event_type: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Render a structured event without replacing ``events.jsonl``."""

        if event_type in _HIDDEN_EVENTS:
            # These events remain available in events.jsonl, but are too noisy
            # for a human-facing progress log.
            return
        message = _format_event(
            event_type,
            payload or {},
            generation_width=self.generation_width,
        )
        if event_type in _ERROR_EVENTS:
            self.error(message)
        elif event_type in _WARNING_EVENTS:
            self.warning(message)
        else:
            self.info(message)

    def close(self) -> None:
        self._close_handlers()

    def _close_handlers(self) -> None:
        for handler in tuple(self._logger.handlers):
            self._logger.removeHandler(handler)
            handler.flush()
            handler.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _format_event(
    event_type: str,
    payload: Mapping[str, Any],
    *,
    generation_width: int = 1,
) -> str:
    if event_type == "generation_started":
        generation = int(payload.get("stage_generation", payload.get("generation", 0)))
        return "{}{} | Started ({})".format(
            _stage_prefix(payload),
            _generation_label(payload, generation, generation_width),
            payload.get("candidate_count", "?"),
        )
    if event_type == "generation_completed":
        generation = int(payload.get("stage_generation", payload.get("generation", 0)))
        message = "{}{} | Completed ({})".format(
            _stage_prefix(payload),
            _generation_label(payload, generation, generation_width),
            payload.get("unique_evaluations", "?"),
        )
        best_metrics = payload.get("best_metrics")
        if isinstance(best_metrics, Mapping):
            primary = str(payload.get("primary_objective", ""))
            for metric, value in best_metrics.items():
                label = _objective_log_label(str(metric))
                if str(metric) == primary:
                    label = "best_" + label
                message += " {}={}".format(label, _format_value(float(value)))
        return message
    if event_type == "base_model_loading_started":
        fields = _pick(payload, ("backend", "engine_mode", "devices"))
        return "Base model | Loading started{}".format(
            _render_fields(fields)
        )
    if event_type == "base_model_loading_completed":
        fields = _pick(
            payload,
            ("elapsed_seconds", "worker_count", "evaluator"),
        )
        return "Base model | Loading completed{}".format(
            _render_fields(fields)
        )
    if event_type == "candidate_completed":
        return _format_candidate_completed(payload)
    if event_type == "objective_best":
        return _format_objective_best(payload, generation_width)
    if event_type == "adapter_pool_release":
        fields = _pick(
            payload,
            (
                "released_source_count",
                "resident_source_count",
                "released_source_bytes",
                "resident_source_bytes",
                "worker_count",
            ),
        )
        state = "Released" if payload.get("supported") else "Skipped"
        return "Adapter pool | {}{}".format(state, _render_fields(fields))
    if event_type == "search_stage_started":
        fields = _pick(
            payload,
            (
                "stage",
                "population_size",
                "generations",
                "dimension",
                "group_count",
                "initial_genome_count",
                "projected_seed_count",
                "inherited_population_count",
                "fitness_inherited",
            ),
        )
        return "STAGE {}/{} | Started{}".format(
            int(payload.get("stage_index", 0)) + 1,
            payload.get("stage_count", "?"),
            _render_fields(fields),
        )
    if event_type == "search_stage_completed":
        fields = _pick(
            payload,
            (
                "stage",
                "group_count",
                "stage_unique_evaluations",
                "pareto_size",
                "promotion_reason",
            ),
        )
        return "STAGE {}/{} | Completed{}".format(
            int(payload.get("stage_index", 0)) + 1,
            payload.get("stage_count", "?"),
            _render_fields(fields),
        )
    if event_type == "resolution_checkpoint":
        fields = _pick(
            payload,
            (
                "fe_total",
                "stage",
                "stage_fe",
                "group_count",
                "decision_dim",
                "directional_coherence",
                "valid_direction_sectors",
                "total_sectors",
                "valid_sector_fraction",
                "marginal_utility",
                "utility_ema",
                "utility_peak",
                "utility_ratio",
                "directional_saturated",
                "utility_saturated",
                "saturation_streak",
                "zero_progress",
                "remaining_fe",
                "future_reserve",
                "slack_fe",
                "promotion_requested",
                "promotion_reason",
            ),
        )
        return "Resolution checkpoint{}".format(_render_fields(fields))
    if event_type == "resolution_promoted":
        fields = _pick(
            payload,
            (
                "old_stage",
                "new_stage",
                "old_group_count",
                "new_group_count",
                "old_dimension",
                "new_dimension",
                "fe_at_promotion",
                "reason",
            ),
        )
        return "Resolution promoted{}".format(_render_fields(fields))
    if event_type in (
        "resolution_symmetry_released",
        "resolution_symmetry_warning",
    ):
        fields = _pick(
            payload,
            ("stage", "batches_observed", "message"),
        )
        return "Resolution symmetry{}".format(_render_fields(fields))
    if event_type == "resolution_trajectory":
        stages = payload.get("stages")
        count = len(stages) if isinstance(stages, list) else 0
        return "Resolution trajectory | Completed ({} stages)".format(count)
    fields = _event_fields(event_type, payload)
    if not fields:
        return event_type
    return "{} | {}".format(event_type, _render_fields(fields).lstrip())


def _format_candidate_completed(payload: Mapping[str, Any]) -> str:
    metrics = payload.get("metrics")
    runtime = payload.get("runtime")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    runtime = runtime if isinstance(runtime, Mapping) else {}
    plan_id = str(payload.get("plan_id", "?"))[:12]
    fields = {
        "quality": _rounded(metrics.get("quality")),
        "worst_group": _rounded(metrics.get("worst_group_quality")),
        "latency_s": _rounded(metrics.get("latency_seconds")),
        "tokens": _rounded(metrics.get("total_tokens")),
        "gpu": runtime.get("gpu_id"),
    }
    fields = {key: value for key, value in fields.items() if value is not None}
    return "CAND {} | Completed{}".format(
        plan_id, _render_fields(fields)
    )


def _format_objective_best(
    payload: Mapping[str, Any], generation_width: int
) -> str:
    label = str(payload.get("objective_label", payload.get("objective", "?")))
    width = max(len(label), int(payload.get("objective_width", len(label))))
    prefix = ""
    scope = payload.get("scope")
    if scope in ("generation", "stage"):
        prefix = _stage_prefix(payload)
    if scope == "generation":
        generation = int(
            payload.get("stage_generation", payload.get("generation", 0))
        )
        prefix += _generation_label(payload, generation, generation_width) + " | "

    parts = [
        "{}OBJ: {:<{}}".format(prefix, label, width),
        "Best CAND: {}".format(str(payload.get("candidate_id", "?"))[:12]),
    ]
    worst = payload.get("worst_group_quality")
    quality = payload.get("quality")
    if worst is not None and quality is not None:
        parts.append(
            "Quality: [{}, {}]".format(
                _format_value(float(worst)), _format_value(float(quality))
            )
        )
    elif quality is not None:
        parts.append("Quality: {}".format(_format_value(float(quality))))
    if payload.get("latency_seconds") is not None:
        parts.append(
            "Latency: {} s".format(
                _format_value(float(payload["latency_seconds"]))
            )
        )
    if payload.get("total_tokens") is not None:
        parts.append(
            "Tokens: {}".format(_format_value(float(payload["total_tokens"])))
        )
    if payload.get("device") is not None:
        parts.append("Device: {}".format(payload["device"]))
    return " | ".join(parts)


def _objective_log_label(metric: str) -> str:
    aliases = {
        "worst_group_quality": "worst_group",
        "fusion_rank_proxy": "rank",
        "active_adapters": "active",
    }
    if metric.startswith("task_quality."):
        return metric[len("task_quality.") :]
    if metric.startswith("domain_quality."):
        return metric[len("domain_quality.") :]
    return aliases.get(metric, metric)


def _stage_prefix(payload: Mapping[str, Any]) -> str:
    stage = payload.get("stage_index")
    if stage is None:
        return ""
    stage_count = payload.get("stage_count")
    if stage_count is None:
        return "Stage {} | ".format(int(stage) + 1)
    width = len(str(int(stage_count)))
    return "Stage {:>{}}/{} | ".format(int(stage) + 1, width, int(stage_count))


def _generation_label(
    payload: Mapping[str, Any], generation: int, fallback_width: int
) -> str:
    # Optimizers and structured events keep the conventional 0-based index;
    # human-facing progress is 1-based so the final batch reads N/N.
    display_generation = int(generation) + 1
    generation_count = payload.get("generation_count")
    if generation_count is None:
        return "GEN {:>{}}".format(display_generation, fallback_width)
    total = int(generation_count)
    width = len(str(total))
    return "GEN {:>{}}/{}".format(display_generation, width, total)


def _render_fields(fields: Mapping[str, Any]) -> str:
    if not fields:
        return ""
    return " " + " ".join(
        "{}={}".format(key, _format_value(value))
        for key, value in fields.items()
    )


def _rounded(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return value


def _event_fields(
    event_type: str, payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Select compact scalar fields for readable progress messages."""

    if event_type == "run_started":
        keys = (
            "experiment",
            "algorithm",
            "runtime_mode",
            "devices",
            "granularities",
            "populations",
            "generations",
            "total_budget",
            "min_stage_fe",
        )
        return _pick(payload, keys)
    if event_type == "evaluator_ready":
        workers = payload.get("workers")
        return _pick(
            payload,
            ("evaluator", "mode", "configured_gpu_ids", "start_method"),
            extra={"worker_count": len(workers) if isinstance(workers, list) else None},
        )
    if event_type == "search_validated":
        return _pick(
            payload,
            ("algorithm", "adapter_pool_mode", "adapter_count", "transition"),
        )
    if event_type == "evaluation_batch_failed":
        return _pick(payload, ("error_type", "message"))
    if event_type == "run_failed":
        return _pick(payload, ("error_type", "message"))
    if event_type == "admission_started":
        return _pick(payload, ("variant_count", "role_wise", "devices"))
    if event_type == "admission_completed":
        return _pick(payload, ("base_quality", "best_variant"))
    return dict(payload)


def _pick(
    payload: Mapping[str, Any],
    keys: tuple,
    extra: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    selected = {
        key: payload[key]
        for key in keys
        if key in payload and payload[key] is not None
    }
    if extra:
        selected.update(
            (key, value)
            for key, value in extra.items()
            if value is not None
        )
    return selected


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return "{:.3f}".format(value)
    if isinstance(value, str) and value and not any(
        character.isspace() for character in value
    ):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def _slug(value: str) -> str:
    return "".join(
        character.lower()
        if character.isalnum() or character in ("-", "_")
        else "-"
        for character in value.strip()
    ).strip("-") or "experiment"


__all__ = ["ExperimentLogger"]
