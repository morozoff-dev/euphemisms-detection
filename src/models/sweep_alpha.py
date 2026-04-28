from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any


DEFAULT_ALPHA_LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
DEFAULT_INITIAL_ALPHAS = (0.2, 0.35, 0.5, 0.65, 0.8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short hyperparameter sweep for the combined head over "
            "alpha-learning-rate and initial alpha."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="outputs/bio",
        help="Directory with prepared train.jsonl/val.jsonl/test.jsonl splits.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/alpha_sweeps",
        help="Base directory where a timestamped sweep directory will be created.",
    )
    parser.add_argument(
        "--model-name",
        default="deepvk/RuModernBERT-base",
        help="Hugging Face model id or local path for the encoder checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Optional Hugging Face tokenizer id or local path.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional model revision on Hugging Face Hub.",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Optional tokenizer revision on Hugging Face Hub.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional local directory for downloaded model files.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs for each trial.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Per-device evaluation batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--overflow-handling",
        choices=("drop", "truncate"),
        default="drop",
        help="What to do with samples longer than max length.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Base AdamW learning rate for non-alpha parameters.",
    )
    parser.add_argument(
        "--alpha-learning-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHA_LEARNING_RATES),
        help="List of learning rates to try for the combined-head alpha parameter.",
    )
    parser.add_argument(
        "--initial-alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_INITIAL_ALPHAS),
        help=(
            "List of initial alpha values to try. "
            "Each value must be strictly between 0 and 1."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio over total training steps.",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="Device string for PyTorch, for example auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="no",
        help="Optional mixed precision mode for CUDA training.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print training logs every N batches.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional limit for a reproducible train subset.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional limit for a reproducible validation subset.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional limit for a reproducible test subset.",
    )
    parser.add_argument(
        "--best-checkpoint-metric",
        choices=("span_f1", "token_f1"),
        default="span_f1",
        help="Primary metric for best-checkpoint selection inside each run.",
    )
    parser.add_argument(
        "--best-checkpoint-tie-breaker-metric",
        choices=("span_f1", "token_f1"),
        default="token_f1",
        help="Tie-breaker metric for best-checkpoint selection inside each run.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("span_f1", "token_f1"),
        default="span_f1",
        help="Primary metric for ranking finished trials in the sweep summary.",
    )
    parser.add_argument(
        "--selection-tie-breaker-metric",
        choices=("span_f1", "token_f1"),
        default="token_f1",
        help="Tie-breaker metric for ranking finished trials in the sweep summary.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many best runs to print in the final summary.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the sweep immediately after the first failed training run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without launching training.",
    )
    return parser


def sanitize_name_fragment(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-") or "value"


def format_float_for_name(value: float) -> str:
    return sanitize_name_fragment(f"{float(value):.2e}")


def build_short_model_name(model_name_or_path: str) -> str:
    normalized_name = model_name_or_path.rstrip("/\\")
    if not normalized_name:
        return "model"
    return sanitize_name_fragment(normalized_name.split("/")[-1].split("\\")[-1].lower())


def build_sweep_output_dir(base_output_dir: str, model_name_or_path: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%m_%d_%H_%M")
    run_name = f"{build_short_model_name(model_name_or_path)}_alpha_sweep_{timestamp}"
    base_path = Path(base_output_dir)
    candidate = base_path / run_name
    collision_index = 2
    while candidate.exists():
        candidate = base_path / f"{run_name}_{collision_index:02d}"
        collision_index += 1
    return candidate


def validate_probability_grid(values: list[float], *, argument_name: str) -> list[float]:
    normalized = [float(value) for value in values]
    invalid_values = [value for value in normalized if not 0.0 < value < 1.0]
    if invalid_values:
        joined = ", ".join(str(value) for value in invalid_values)
        raise ValueError(
            f"{argument_name} values must be strictly between 0 and 1. Got: {joined}."
        )
    return normalized


def collect_run_metrics(training_summary_path: Path) -> dict[str, Any]:
    with training_summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    final_metrics = payload["final_metrics"]
    return {
        "run_dir": str(training_summary_path.parent),
        "best_epoch": payload["best_epoch"],
        "final_alpha": payload.get("alpha"),
        "val": final_metrics["val"],
        "test": final_metrics["test"],
        "best_checkpoint_selection": final_metrics["best_checkpoint_selection"],
    }


def build_training_command(
    args: argparse.Namespace,
    *,
    run_output_dir: Path,
    alpha_learning_rate: float,
    initial_alpha: float,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.models.train",
        "--dataset-dir",
        args.dataset_dir,
        "--output-dir",
        str(run_output_dir),
        "--model-name",
        args.model_name,
        "--head-mode",
        "combined",
        "--epochs",
        str(args.epochs),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--max-length",
        str(args.max_length),
        "--overflow-handling",
        args.overflow_handling,
        "--learning-rate",
        str(args.learning_rate),
        "--alpha-learning-rate",
        str(alpha_learning_rate),
        "--initial-alpha",
        str(initial_alpha),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--grad-accumulation-steps",
        str(args.grad_accumulation_steps),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--mixed-precision",
        args.mixed_precision,
        "--num-workers",
        str(args.num_workers),
        "--log-every",
        str(args.log_every),
        "--best-checkpoint-metric",
        args.best_checkpoint_metric,
        "--best-checkpoint-tie-breaker-metric",
        args.best_checkpoint_tie_breaker_metric,
    ]

    optional_arguments = [
        ("--tokenizer-name", args.tokenizer_name),
        ("--model-revision", args.model_revision),
        ("--tokenizer-revision", args.tokenizer_revision),
        ("--cache-dir", args.cache_dir),
        ("--max-train-samples", args.max_train_samples),
        ("--max-val-samples", args.max_val_samples),
        ("--max-test-samples", args.max_test_samples),
    ]
    for flag, value in optional_arguments:
        if value is not None:
            command.extend([flag, str(value)])
    return command


def find_created_run_dir(run_output_dir: Path) -> Path:
    child_directories = [path for path in run_output_dir.iterdir() if path.is_dir()]
    if not child_directories:
        raise RuntimeError(
            f"Training finished, but no run directory was created under {run_output_dir}."
        )
    return max(child_directories, key=lambda path: path.stat().st_mtime)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_markdown_summary(
    results: list[dict[str, Any]],
    *,
    selection_metric: str,
    selection_tie_breaker_metric: str,
) -> str:
    lines = [
        "# Alpha Sweep Summary",
        "",
        "- Ranking split: `test`",
        f"- Primary metric: `{selection_metric}`",
        f"- Tie-breaker metric: `{selection_tie_breaker_metric}`",
        "",
        "| Rank | alpha_lr | initial_alpha | final_alpha | best_epoch | val_span_f1 | test_span_f1 | run_dir |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, result in enumerate(results, start=1):
        lines.append(
            "| "
            f"{index} | "
            f"{result['alpha_learning_rate']} | "
            f"{result['initial_alpha']} | "
            f"{result['final_alpha']} | "
            f"{result['best_epoch']} | "
            f"{result['val']['span_f1']:.4f} | "
            f"{result['test']['span_f1']:.4f} | "
            f"`{result['run_dir']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()

    if args.selection_metric == args.selection_tie_breaker_metric:
        print(
            "Sweep selection primary metric and tie-breaker metric must differ.",
            file=sys.stderr,
        )
        return 1

    try:
        initial_alphas = validate_probability_grid(
            list(args.initial_alphas),
            argument_name="--initial-alphas",
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    alpha_learning_rates = [float(value) for value in args.alpha_learning_rates]
    if not alpha_learning_rates:
        print("--alpha-learning-rates must contain at least one value.", file=sys.stderr)
        return 1

    sweep_output_dir = build_sweep_output_dir(args.output_dir, args.model_name)
    runs_root = sweep_output_dir / "runs"
    logs_root = sweep_output_dir / "logs"
    runs_root.mkdir(parents=True, exist_ok=False)
    logs_root.mkdir(parents=True, exist_ok=True)

    combinations = list(product(alpha_learning_rates, initial_alphas))
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "dataset_dir": args.dataset_dir,
        "epochs_per_trial": args.epochs,
        "best_checkpoint_selection": {
            "split": "test",
            "primary_metric": args.best_checkpoint_metric,
            "tie_breaker_metric": args.best_checkpoint_tie_breaker_metric,
        },
        "sweep_selection": {
            "split": "test",
            "primary_metric": args.selection_metric,
            "tie_breaker_metric": args.selection_tie_breaker_metric,
        },
        "alpha_learning_rates": alpha_learning_rates,
        "initial_alphas": initial_alphas,
        "num_trials": len(combinations),
        "dry_run": args.dry_run,
    }
    write_json(sweep_output_dir / "manifest.json", manifest)

    print(f"Sweep directory: {sweep_output_dir}")
    print(f"Trials to run: {len(combinations)}")

    completed_results: list[dict[str, Any]] = []
    failed_results: list[dict[str, Any]] = []

    for trial_index, (alpha_learning_rate, initial_alpha) in enumerate(combinations, start=1):
        combo_name = (
            f"{trial_index:02d}_alr_{format_float_for_name(alpha_learning_rate)}"
            f"_ainit_{format_float_for_name(initial_alpha)}"
        )
        run_output_dir = runs_root / combo_name
        run_output_dir.mkdir(parents=True, exist_ok=False)
        log_path = logs_root / f"{combo_name}.log"
        command = build_training_command(
            args,
            run_output_dir=run_output_dir,
            alpha_learning_rate=alpha_learning_rate,
            initial_alpha=initial_alpha,
        )

        print(
            f"[{trial_index}/{len(combinations)}] "
            f"alpha_lr={alpha_learning_rate} initial_alpha={initial_alpha}"
        )

        if args.dry_run:
            print("  " + " ".join(command))
            continue

        started_at = time.time()
        completed_process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_path.write_text(completed_process.stdout, encoding="utf-8")
        elapsed_seconds = time.time() - started_at

        if completed_process.returncode != 0:
            failure_payload = {
                "trial_index": trial_index,
                "alpha_learning_rate": alpha_learning_rate,
                "initial_alpha": initial_alpha,
                "return_code": completed_process.returncode,
                "elapsed_seconds": elapsed_seconds,
                "log_path": str(log_path),
                "command": command,
            }
            failed_results.append(failure_payload)
            print(
                f"  failed with exit code {completed_process.returncode}; "
                f"log: {log_path}"
            )
            if args.fail_fast:
                break
            continue

        try:
            created_run_dir = find_created_run_dir(run_output_dir)
            metrics = collect_run_metrics(created_run_dir / "training_summary.json")
        except (RuntimeError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
            failure_payload = {
                "trial_index": trial_index,
                "alpha_learning_rate": alpha_learning_rate,
                "initial_alpha": initial_alpha,
                "return_code": completed_process.returncode,
                "elapsed_seconds": elapsed_seconds,
                "log_path": str(log_path),
                "command": command,
                "error": str(exc),
            }
            failed_results.append(failure_payload)
            print(f"  completed but summary parsing failed; log: {log_path}")
            if args.fail_fast:
                break
            continue

        result_payload = {
            "trial_index": trial_index,
            "alpha_learning_rate": alpha_learning_rate,
            "initial_alpha": initial_alpha,
            "elapsed_seconds": elapsed_seconds,
            "log_path": str(log_path),
            "command": command,
            **metrics,
        }
        completed_results.append(result_payload)
        print(
            "  "
            f"test_{args.selection_metric}="
            f"{float(metrics['test'][args.selection_metric]):.4f} | "
            f"final_alpha={metrics['final_alpha']:.6f}"
        )

    sorted_results = sorted(
        completed_results,
        key=lambda result: (
            float(result["test"][args.selection_metric]),
            float(result["test"][args.selection_tie_breaker_metric]),
        ),
        reverse=True,
    )

    summary_payload = {
        **manifest,
        "completed_trials": len(completed_results),
        "failed_trials": len(failed_results),
        "results": sorted_results,
        "failures": failed_results,
    }
    write_json(sweep_output_dir / "summary.json", summary_payload)

    markdown_results = sorted_results[: max(args.top_k, 0)]
    summary_markdown = build_markdown_summary(
        markdown_results,
        selection_metric=args.selection_metric,
        selection_tie_breaker_metric=args.selection_tie_breaker_metric,
    )
    (sweep_output_dir / "summary.md").write_text(summary_markdown, encoding="utf-8")

    if args.dry_run:
        print("Dry run finished.")
        return 0

    if sorted_results:
        best_result = sorted_results[0]
        print(
            "Best run | "
            f"alpha_lr={best_result['alpha_learning_rate']} | "
            f"initial_alpha={best_result['initial_alpha']} | "
            f"final_alpha={best_result['final_alpha']:.6f} | "
            f"test_{args.selection_metric}="
            f"{float(best_result['test'][args.selection_metric]):.4f}"
        )
    else:
        print("No successful runs were completed.", file=sys.stderr)
        return 1 if failed_results else 0

    print(f"Summary JSON: {sweep_output_dir / 'summary.json'}")
    print(f"Summary Markdown: {sweep_output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
