"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .backends.audit import PeftAdapterPoolAuditor
from .backends import OptionalDependencyError
from .core.config import (
    ConfigError,
    apply_experiment_overrides,
    load_experiment_config,
    resolve_device_ids,
)
from .runtime.experiment import ComponentFactory, ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentmerge",
        description="Search role-specific adapter mixtures for a fixed agent system.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Parse and validate an experiment YAML/JSON file."
    )
    validate_parser.add_argument("config")
    _add_override_arguments(validate_parser)

    search_parser = subparsers.add_parser(
        "search", help="Run a configured multi-objective search."
    )
    search_parser.add_argument("config")
    _add_override_arguments(search_parser)

    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit PEFT adapter configs without loading base-model weights.",
    )
    audit_parser.add_argument("config")

    smoke_parser = subparsers.add_parser(
        "smoke-real",
        help="Run base, one-hot, and on-demand fusion on one real GPU backend.",
    )
    smoke_parser.add_argument("config")
    smoke_parser.add_argument("--device", dest="smoke_device", default="0")
    smoke_parser.add_argument("--max-new-tokens", type=int, default=16)
    smoke_parser.add_argument("--one-hot-limit", type=int, default=1)
    smoke_parser.add_argument(
        "--prompt",
        default="What is 2 + 2? Reply with only the answer.",
    )

    admission_parser = subparsers.add_parser(
        "admit-adapters",
        help="Evaluate base and deterministic one-hot adapter variants.",
    )
    admission_parser.add_argument("config")
    admission_parser.add_argument("--role-wise", action="store_true")
    _add_override_arguments(admission_parser)

    replay_parser = subparsers.add_parser(
        "replay-candidate",
        help="Replay one saved global candidate on a single real backend.",
    )
    replay_parser.add_argument("config")
    replay_parser.add_argument("--candidate", required=True)
    replay_parser.add_argument("--device", dest="replay_device", default="0")
    replay_parser.add_argument("--gpu_memory_utilization", type=float)
    replay_parser.add_argument("--repeats", type=int, default=1)
    replay_parser.add_argument("--max_cases", type=int)
    replay_parser.add_argument("--batch_size", type=int)
    replay_parser.add_argument("--report", required=True)
    return parser


def _add_override_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose search and run controls independently from experiment YAML."""

    parser.add_argument("--run-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--objective",
        dest="objectives",
        action="append",
        metavar="METRIC:DIRECTION",
        help=(
            "Override YAML objectives; repeat for each objective, for example "
            "--objective task_quality.math500:maximize"
        ),
    )
    parser.add_argument("--algorithm", default=None)
    parser.add_argument(
        "--search_granularity",
        nargs="+",
        choices=(
            "global",
            "adapter",
            "layer",
            "layer_group",
            "module_type",
            "layer_module",
        ),
        help=(
            "One or more fusion stages. For example: "
            "--search_granularity global layer"
        ),
    )
    parser.add_argument("--populations", nargs="+", type=int)
    parser.add_argument("--generations", nargs="+", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--runtime-mode", choices=("local", "multi_gpu"))
    parser.add_argument("--device", nargs="+")
    parser.add_argument("--evaluation-timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--crossover_probability", type=float)
    parser.add_argument("--mutation_probability", type=float)
    parser.add_argument("--sparsity_mutation_probability", type=float)
    parser.add_argument("--mutation_sigma", type=float)
    parser.add_argument("--differential_weight", type=float)
    parser.add_argument("--neighborhood_size", type=int)
    parser.add_argument("--neighborhood_mating_probability", type=float)
    parser.add_argument("--max_replacements", type=int)
    parser.add_argument(
        "--decomposition",
        choices=("tchebycheff", "weighted_sum", "pbi"),
    )
    parser.add_argument("--pbi_theta", type=float)
    parser.add_argument("--mutation_distribution_index", type=float)
    parser.add_argument("--refinement_groups", nargs="+", type=int)
    parser.add_argument("--total_budget", type=int)
    parser.add_argument("--min_stage_fe", nargs="+", type=int)
    parser.add_argument("--first_split", type=int)
    parser.add_argument("--num_sectors", type=int)
    parser.add_argument("--sector_seed", type=int)
    parser.add_argument("--direction_window", type=int)
    parser.add_argument("--min_directions_per_sector", type=int)
    parser.add_argument("--min_valid_sector_fraction", type=float)
    parser.add_argument("--direction_norm_eps", type=float)
    parser.add_argument("--coherence_threshold", type=float)
    parser.add_argument("--utility_ratio_threshold", type=float)
    parser.add_argument("--saturation_patience", type=int)
    parser.add_argument("--utility_ema_beta", type=float)
    parser.add_argument("--utility_eps", type=float)
    parser.add_argument("--checkpoint_interval_fe", type=int)
    zero_progress = parser.add_mutually_exclusive_group()
    zero_progress.add_argument(
        "--enable_zero_progress_fallback",
        dest="enable_zero_progress_fallback",
        action="store_true",
    )
    zero_progress.add_argument(
        "--disable_zero_progress_fallback",
        dest="enable_zero_progress_fallback",
        action="store_false",
    )
    parser.set_defaults(enable_zero_progress_fallback=None)
    parser.add_argument("--utility_zero_eps", type=float)
    parser.add_argument("--symmetry_diagnostic_batches", type=int)
    parser.add_argument("--min_active_adapters", type=int)
    parser.add_argument("--max_active_adapters", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--group_size", type=int)
    parser.add_argument("--group_count", type=int)


def _load_resolved_config(args: argparse.Namespace):
    config = load_experiment_config(args.config)
    return apply_experiment_overrides(
        config,
        run_name=getattr(args, "run_name", None),
        output_dir=getattr(args, "output_dir", None),
        seed=getattr(args, "seed", None),
        objectives=getattr(args, "objectives", None),
        algorithm=getattr(args, "algorithm", None),
        search_granularity=getattr(args, "search_granularity", None),
        populations=getattr(args, "populations", None),
        generations=getattr(args, "generations", None),
        workers=getattr(args, "workers", None),
        checkpoint_every=getattr(args, "checkpoint_every", None),
        max_cases=getattr(args, "max_cases", None),
        batch_size=getattr(args, "batch_size", None),
        runtime_mode=getattr(args, "runtime_mode", None),
        device=getattr(args, "device", None),
        evaluation_timeout_seconds=getattr(
            args, "evaluation_timeout_seconds", None
        ),
        max_retries=getattr(args, "max_retries", None),
        crossover_probability=getattr(args, "crossover_probability", None),
        mutation_probability=getattr(args, "mutation_probability", None),
        sparsity_mutation_probability=getattr(
            args, "sparsity_mutation_probability", None
        ),
        mutation_sigma=getattr(args, "mutation_sigma", None),
        differential_weight=getattr(args, "differential_weight", None),
        neighborhood_size=getattr(args, "neighborhood_size", None),
        neighborhood_mating_probability=getattr(
            args, "neighborhood_mating_probability", None
        ),
        max_replacements=getattr(args, "max_replacements", None),
        decomposition=getattr(args, "decomposition", None),
        pbi_theta=getattr(args, "pbi_theta", None),
        mutation_distribution_index=getattr(
            args, "mutation_distribution_index", None
        ),
        refinement_groups=getattr(args, "refinement_groups", None),
        total_budget=getattr(args, "total_budget", None),
        min_stage_fe=getattr(args, "min_stage_fe", None),
        first_split=getattr(args, "first_split", None),
        num_sectors=getattr(args, "num_sectors", None),
        sector_seed=getattr(args, "sector_seed", None),
        direction_window=getattr(args, "direction_window", None),
        min_directions_per_sector=getattr(
            args, "min_directions_per_sector", None
        ),
        min_valid_sector_fraction=getattr(
            args, "min_valid_sector_fraction", None
        ),
        direction_norm_eps=getattr(args, "direction_norm_eps", None),
        coherence_threshold=getattr(args, "coherence_threshold", None),
        utility_ratio_threshold=getattr(
            args, "utility_ratio_threshold", None
        ),
        saturation_patience=getattr(args, "saturation_patience", None),
        utility_ema_beta=getattr(args, "utility_ema_beta", None),
        utility_eps=getattr(args, "utility_eps", None),
        checkpoint_interval_fe=getattr(args, "checkpoint_interval_fe", None),
        enable_zero_progress_fallback=getattr(
            args, "enable_zero_progress_fallback", None
        ),
        utility_zero_eps=getattr(args, "utility_zero_eps", None),
        symmetry_diagnostic_batches=getattr(
            args, "symmetry_diagnostic_batches", None
        ),
        min_active_adapters=getattr(args, "min_active_adapters", None),
        max_active_adapters=getattr(args, "max_active_adapters", None),
        num_layers=getattr(args, "num_layers", None),
        group_size=getattr(args, "group_size", None),
        group_count=getattr(args, "group_count", None),
    )


def main(argv: Sequence[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load_resolved_config(args)
        if args.command == "validate":
            search_description = ComponentFactory().validate_search(config)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "name": config.name,
                        "model": config.model.name_or_path,
                        "base_model_id": config.model.base_model_identity,
                        "agents": list(config.agent_names),
                        "adapters": list(config.adapter_names),
                        "runtime": {
                            "mode": config.runtime.mode,
                            "gpu_ids": list(config.runtime.gpu_ids),
                        },
                        "search": {
                            "algorithm": config.search.algorithm,
                            "granularities": list(config.search.granularities),
                            "stages": search_description.get("stages"),
                        },
                        "objectives": [
                            {
                                "metric": objective.metric,
                                "direction": objective.direction,
                            }
                            for objective in config.objectives
                        ],
                        "available_metrics": search_description.get(
                            "available_objective_metrics", []
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "search":
            summary = ExperimentRunner(config).run()
            print(
                json.dumps(
                    {
                        "run_dir": summary.run_dir,
                        "pareto_size": summary.pareto_size,
                        "unique_evaluations": summary.unique_evaluations,
                        "generations_completed": summary.generations_completed,
                        "objectives": list(summary.objective_names),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "audit":
            report = PeftAdapterPoolAuditor().audit(config)
            print(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
            )
            return 0 if report.passed else 3
        if args.command == "smoke-real":
            # Set visibility before the diagnostic imports or initializes
            # PyTorch. The backend sees the selected physical GPU as cuda:0.
            smoke_device = resolve_device_ids((str(args.smoke_device),))[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = smoke_device
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from .evaluation.diagnostics import BackendSmokeRunner

            report = BackendSmokeRunner(config).run(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                one_hot_limit=args.one_hot_limit,
            )
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            return 0 if report.passed else 4
        if args.command == "admit-adapters":
            from .evaluation.admission import AdapterAdmissionRunner

            summary = AdapterAdmissionRunner(config).run(
                include_rolewise=args.role_wise
            )
            print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
            return 0
        if args.command == "replay-candidate":
            replay_device = resolve_device_ids((str(args.replay_device),))[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = replay_device
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            config = apply_experiment_overrides(
                config,
                max_cases=args.max_cases,
                batch_size=args.batch_size,
                runtime_mode="local",
            )
            from .evaluation.replay import CandidateReplayRunner

            report = CandidateReplayRunner(
                config,
                visible_device=replay_device,
                gpu_memory_utilization=args.gpu_memory_utilization,
            ).run(args.candidate, repeats=args.repeats)
            report_path = Path(args.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    report.to_dict(),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
            print(
                json.dumps(
                    {
                        "passed": report.passed,
                        "report": str(report_path),
                        "device": report.visible_device,
                        "gpu_memory_utilization": (
                            report.gpu_memory_utilization
                        ),
                        "backend_load_seconds": report.backend_load_seconds,
                        "successful_repeats": report.successful_repeats,
                        "repeats_requested": report.repeats_requested,
                        "attempt_seconds": [
                            item.elapsed_seconds for item in report.attempts
                        ],
                        "error": (
                            {
                                "phase": report.error.phase,
                                "error_type": report.error.error_type,
                                "message": report.error.message,
                            }
                            if report.error is not None
                            else (
                                {
                                    "phase": report.attempts[-1].error.phase,
                                    "error_type": (
                                        report.attempts[-1].error.error_type
                                    ),
                                    "message": report.attempts[-1].error.message,
                                }
                                if report.attempts
                                and report.attempts[-1].error is not None
                                else None
                            )
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if report.passed else 5
    except (ConfigError, OptionalDependencyError, ValueError, KeyError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
