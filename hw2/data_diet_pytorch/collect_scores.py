"""Collect every importance score in full-length training runs: python collect_scores.py [--runs 1 2 ... 10]

One run = one independent training of ResNet18 on all of CIFAR-10 (seeds follow the original run_full_data.py).
At the score epochs the run saves, for every training example (original CIFAR-10 order):
    EL2N, L1 error, max/sum margin, loss, entropy, true-class probability, predicted class   (all epochs)
    GraNd                                                                                    (see --grand-*)
    forgetting score / event count / first-learned step                                      (all epochs)
into <run>/scores/epoch_XXX/<name>.npy. Checkpoints are kept too, so any score can be recomputed later.
Completed runs (valid run_manifest.json) are skipped, interrupted ones are moved aside and restarted.
"""
import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from mlflow import MlflowClient
from scipy.stats import spearmanr

from cifar10_data import CLASS_NAMES, load_cifar10
from importance_scores import ForgettingTracker, compute_error_scores, compute_grand_scores
from run_logging import MLFLOW_TRACKING_URI, verify_run_directory
from training_loop import TrainingConfig, train_with_fixed_steps

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "exps" / "score_collection"
SCORE_PAIRS_FOR_CORRELATION = (("el2n_score", "loss"), ("el2n_score", "grand_score"), ("el2n_score", "forgetting_score"),
                               ("grand_score", "forgetting_score"), ("grand_score", "loss"))
MIN_FREE_DISK_GB_PER_RUN = 1.0


def finite_ranks_for_correlation(values: np.ndarray) -> np.ndarray:
    """Forgetting uses +inf for never-learned examples; give them the largest rank for correlations."""
    return np.where(np.isfinite(values), values, np.nanmax(values[np.isfinite(values)], initial=0) + 1)


def describe(name: str, values: np.ndarray) -> dict:
    p10, p50, p90, p99 = np.percentile(values, [10, 50, 90, 99])
    return {f"{name}_mean": values.mean(), f"{name}_std": values.std(), f"{name}_p10": p10, f"{name}_p50": p50,
            f"{name}_p90": p90, f"{name}_p99": p99}


def summarize_scores(scores: dict) -> dict:
    """Distribution statistics and Spearman correlations between score types (logged for every score epoch)."""
    summary = {}
    for name in ("el2n_score", "grand_score", "loss", "entropy"):
        if name in scores:
            summary.update(describe(name, scores[name]))
    finite_forgetting = scores["forgetting_score"][np.isfinite(scores["forgetting_score"])]
    if len(finite_forgetting):
        summary.update(describe("forgetting_score_finite", finite_forgetting))
    for first, second in SCORE_PAIRS_FOR_CORRELATION:
        if first in scores and second in scores:
            a, b = (finite_ranks_for_correlation(scores[name]) for name in (first, second))
            if a.std() > 0 and b.std() > 0:
                summary[f"spearman_{first}_vs_{second}"] = spearmanr(a, b).statistic
    return summary


class ScoreCollector:
    """training_loop score_callback: computes and saves all scores of the current model."""

    def __init__(self, data, tracker: ForgettingTracker, grand_epochs):
        self.data, self.tracker, self.grand_epochs = data, tracker, set(grand_epochs)

    def __call__(self, epoch: int, model, recorder) -> dict:
        start = time.perf_counter()
        scores = compute_error_scores(model, self.data.train_images, self.data.train_labels)
        if epoch in self.grand_epochs:
            scores["grand_score"] = compute_grand_scores(model, self.data.train_images, self.data.train_labels)
        scores.update(self.tracker.snapshot())
        folder = f"scores/epoch_{epoch:03d}"
        (recorder.run_directory / folder).mkdir(parents=True, exist_ok=True)
        labels_file = "scores/train_labels.npy"
        if not (recorder.run_directory / labels_file).exists():
            np.save(recorder.run_directory / labels_file, self.data.train_labels.cpu().numpy().astype(np.int8))
            recorder.expect_file(labels_file)
        for name, values in scores.items():
            np.save(recorder.run_directory / folder / f"{name}.npy", values)
            recorder.expect_file(f"{folder}/{name}.npy")
        summary = summarize_scores(scores)
        summary["scoring_seconds"] = time.perf_counter() - start
        return {key: float(value) for key, value in summary.items()}


def sanity_report(run_directory: Path, labels: np.ndarray, reference_epoch: int) -> tuple:
    """Cheap automatic checks of a finished run. Returns (problems, metrics for mlflow)."""
    folder = run_directory / "scores" / f"epoch_{reference_epoch:03d}"
    el2n, loss = np.load(folder / "el2n_score.npy"), np.load(folder / "loss.npy")
    problems, metrics = [], {}
    if not np.isfinite(el2n).all() or el2n.min() < 0 or el2n.max() > np.sqrt(2) + 1e-4:
        problems.append("EL2N has non-finite values or leaves [0, sqrt(2)]")
    correlation = spearmanr(el2n, loss).statistic
    metrics["sanity_spearman_el2n_vs_loss"] = correlation
    if correlation < 0.8:
        problems.append(f"EL2N and loss disagree (Spearman {correlation:.2f})")
    hardest_half = np.argsort(el2n)[len(el2n) // 2:]
    counts = np.bincount(labels[hardest_half], minlength=len(CLASS_NAMES))
    metrics.update({f"hardest_half_el2n_count_{name}": float(count) for name, count in zip(CLASS_NAMES, counts)})
    if counts[CLASS_NAMES.index("cat")] <= counts[CLASS_NAMES.index("automobile")]:
        print("  WARNING: the hardest half by EL2N has no skew towards cats over automobiles")
    return problems, metrics


def log_after_training(experiment: str, run_name: str, run_directory: Path, upload_scores: bool, metrics: dict) -> None:
    """The training run is closed when train_with_fixed_steps returns; attach extra artifacts and metrics afterwards."""
    client = MlflowClient(MLFLOW_TRACKING_URI)
    found = client.search_runs([client.get_experiment_by_name(experiment).experiment_id],
                               filter_string=f"tags.mlflow.runName = '{run_name}'",
                               order_by=["attributes.start_time DESC"], max_results=1)
    if not found:
        print(f"  WARNING: mlflow run '{run_name}' not found, extra logging skipped")
        return
    run_id = found[0].info.run_id
    for key, value in metrics.items():
        client.log_metric(run_id, key, float(value))
    if upload_scores:
        client.log_artifacts(run_id, str(run_directory / "scores"), "scores")
        client.log_artifact(run_id, str(run_directory / "run_manifest.json"), "run_files")


def prepare_run_directory(run_directory: Path) -> bool:
    """True if the run must be (re)done. Completed runs are kept, interrupted ones are moved aside, never deleted."""
    if not run_directory.exists():
        return True
    if not verify_run_directory(run_directory, check_hashes=True):
        return False
    moved = run_directory.with_name(f"{run_directory.name}_incomplete_{datetime.now():%Y%m%d_%H%M%S}")
    run_directory.rename(moved)
    print(f"  found an incomplete {run_directory.name}, moved to {moved.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--total-epochs", type=int, default=200)
    parser.add_argument("--stop-after-epochs", type=int, default=None, help="for smoke tests only")
    parser.add_argument("--score-epochs", type=int, nargs="+", default=[0, 4, 8, 10, 12, 16, 20, 50, 100, 200])
    parser.add_argument("--grand-epochs-full", type=int, nargs="+", default=[0, 4, 8, 10, 12, 16, 20])
    parser.add_argument("--runs-with-full-grand", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--grand-epochs-other", type=int, nargs="+", default=[20])
    parser.add_argument("--precision", default="bf16", choices=["bf16", "tf32", "fp32"])
    parser.add_argument("--mlflow-experiment", default="data-diet-score-collection")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    experiment = None if args.no_mlflow else args.mlflow_experiment
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs_to_do = [run for run in args.runs if prepare_run_directory(args.output_root / f"run_{run:02d}")]
    free_gb = shutil.disk_usage(args.output_root).free / 2**30
    print(f"runs to do: {runs_to_do} (already complete: {sorted(set(args.runs) - set(runs_to_do))}); "
          f"free disk {free_gb:.1f} GB, need ~{MIN_FREE_DISK_GB_PER_RUN * len(runs_to_do):.0f} GB")
    if free_gb < MIN_FREE_DISK_GB_PER_RUN * len(runs_to_do) + 2:
        print("not enough free disk space")
        return 2
    data = load_cifar10()
    labels = data.train_labels.cpu().numpy()
    last_epoch = min(args.total_epochs, args.stop_after_epochs or args.total_epochs)
    reference_epoch = max(epoch for epoch in args.score_epochs if epoch <= min(20, last_epoch))
    failed, summary_lines = [], []
    for run in runs_to_do:
        run_directory = args.output_root / f"run_{run:02d}"
        run_name = f"score_run_{run:02d}"
        grand_epochs = args.grand_epochs_full if run in args.runs_with_full_grand else args.grand_epochs_other
        print(f"\n=== {run_name}: scores at epochs {args.score_epochs}, GraNd at {grand_epochs}", flush=True)
        tracker = ForgettingTracker(data.num_train_examples)
        config = TrainingConfig(run_number=run, total_epochs=args.total_epochs, stop_after_epochs=args.stop_after_epochs,
                                precision=args.precision, checkpoint_epochs=args.score_epochs,
                                score_epochs=args.score_epochs,
                                extra_parameters={"purpose": "score_collection", "grand_epochs": grand_epochs})
        started = time.perf_counter()
        result = train_with_fixed_steps(
            data, config, run_directory, step_observers=[tracker],
            score_callback=ScoreCollector(data, tracker, grand_epochs), mlflow_experiment=experiment,
            mlflow_run_name=run_name, mlflow_tags={"stage": "score_collection", "run_number": str(run)})
        problems = verify_run_directory(run_directory, check_hashes=True)
        sanity_problems, sanity_metrics = sanity_report(run_directory, labels, reference_epoch)
        final = result.epoch_rows[-1]
        if experiment:
            log_after_training(experiment, run_name, run_directory, run in args.runs_with_full_grand,
                               sanity_metrics)
        all_problems = problems + sanity_problems
        if all_problems:
            failed.append(run)
        summary_lines.append(f"{run_name}: final test acc {final['test_accuracy']:.4f}, "
                             f"{(time.perf_counter() - started) / 60:.1f} min, "
                             f"{'OK' if not all_problems else 'PROBLEMS: ' + '; '.join(all_problems)}")
        print("  " + summary_lines[-1], flush=True)
    print("\n=== summary")
    print("\n".join(summary_lines) or "nothing to do")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
