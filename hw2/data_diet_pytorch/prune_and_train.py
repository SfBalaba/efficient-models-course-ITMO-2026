"""Prune the training set by an importance score and train on the rest.

    python prune_and_train.py --score-type el2n_score --prune-fraction 0.5 --seed-run 101

The examples with the LOWEST score are removed (as in the paper), so the kept subset has the highest scores. Random
pruning of the same size is the control (--score-type random). The number of training steps stays 78 000 whatever the
subset size. The score is read from the average over the score-collection runs (aggregate_scores.py), taken at
--score-epoch (20 in the paper); any other score type or epoch works without code changes (grand_score,
forgetting_score, loss, ...). Training seeds (--seed-run) must differ from the seeds used to collect the scores (1-10).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient

from cifar10_data import CLASS_NAMES, Cifar10Data, load_cifar10
from run_logging import MLFLOW_TRACKING_URI, build_run_manifest, sha256_of_file, verify_run_directory
from training_loop import TrainingConfig, train_with_fixed_steps

PRUNING_ROOT = Path(__file__).resolve().parent / "exps" / "pruning"
AVERAGED_SCORES_ROOT = Path(__file__).resolve().parent / "exps" / "score_collection" / "aggregated"
RANDOM_SCORE_TYPE = "random"
RANDOM_SUBSET_SEED_OFFSET = 7000
LAST_SEED_RUN_USED_FOR_SCORES = 10


def run_name_for(score_type: str, prune_fraction: float, seed_run: int) -> str:
    return f"{score_type}_prune{round(prune_fraction * 100):02d}_seed{seed_run:03d}"


def run_directory_for(output_root: Path, score_type: str, prune_fraction: float, seed_run: int) -> Path:
    return output_root / score_type / f"prune_{round(prune_fraction * 100):02d}" / f"seed_{seed_run:03d}"


def select_examples(score_type: str, scores: Optional[np.ndarray], prune_fraction: float, subset_seed: int,
                    num_examples: int) -> np.ndarray:
    """Indices of the kept examples (sorted). Ties between equal scores are broken by index (stable sort)."""
    kept_count = num_examples - int(round(prune_fraction * num_examples))
    if score_type == RANDOM_SCORE_TYPE:
        chosen = np.random.RandomState(subset_seed).choice(num_examples, kept_count, replace=False)
    else:
        chosen = np.argsort(scores, kind="stable")[num_examples - kept_count:]
    return np.sort(chosen).astype(np.int32)


def describe_selection(kept: np.ndarray, labels: np.ndarray, scores: Optional[np.ndarray], scores_file: Optional[Path],
                       score_type: str, score_epoch: int, prune_fraction: float) -> dict:
    counts = np.bincount(labels[kept], minlength=len(CLASS_NAMES))
    summary = {"score_type": score_type, "score_epoch": score_epoch, "requested_prune_fraction": prune_fraction,
               "kept_examples": int(len(kept)), "kept_per_class": {n: int(c) for n, c in zip(CLASS_NAMES, counts)}}
    if scores is not None:
        kept_scores = scores[kept]
        removed = np.setdiff1d(np.arange(len(scores)), kept)
        summary.update(scores_file=str(scores_file), scores_file_sha256=sha256_of_file(scores_file),
                       cutoff_score_lowest_kept=float(kept_scores.min()),
                       highest_removed_score=float(scores[removed].max()) if len(removed) else None,
                       kept_scores_infinite=int(np.isinf(kept_scores).sum()))
    return summary


def extend_manifest(run_directory: Path, extra_files: list) -> None:
    """Add files written after training to the run manifest (same checks and hashes as the training files)."""
    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    build_run_manifest(run_directory, set(manifest["files"]) | set(extra_files))


def log_selection_to_mlflow(experiment: str, run_name: str, run_directory: Path, summary: dict) -> None:
    client = MlflowClient(MLFLOW_TRACKING_URI)
    found = client.search_runs([client.get_experiment_by_name(experiment).experiment_id],
                               filter_string=f"tags.mlflow.runName = '{run_name}'",
                               order_by=["attributes.start_time DESC"], max_results=1)
    if not found:
        print(f"  WARNING: mlflow run '{run_name}' not found, selection details not logged")
        return
    run_id = found[0].info.run_id
    for name, count in summary["kept_per_class"].items():
        client.log_metric(run_id, f"kept_count_{name}", count)
        client.log_metric(run_id, f"kept_share_{name}", count / summary["kept_examples"])
    if summary.get("cutoff_score_lowest_kept") is not None and np.isfinite(summary["cutoff_score_lowest_kept"]):
        client.log_metric(run_id, "cutoff_score_lowest_kept", summary["cutoff_score_lowest_kept"])
    for file_name in ("selection_summary.json", "kept_indices.npy", "run_manifest.json"):
        client.log_artifact(run_id, str(run_directory / file_name), "run_files")


def result_row_from_directory(run_directory: Path, score_type: str, prune_fraction: float, seed_run: int,
                              reused_finished_run: bool) -> dict:
    table = pd.read_csv(run_directory / "epoch_metrics.csv")
    final = table.iloc[-1]
    selection = json.loads((run_directory / "selection_summary.json").read_text())
    return {"score_type": score_type, "prune_fraction": prune_fraction, "seed_run": seed_run,
            "kept_examples": selection["kept_examples"], "keep_fraction": selection["kept_examples"] / 50000,
            "final_test_accuracy": float(final["test_accuracy"]), "best_test_accuracy": float(table["test_accuracy"].max()),
            "train_minutes": float(final["elapsed_minutes"]), "reused_finished_run": reused_finished_run,
            "run_directory": str(run_directory)}


def run_prune_and_train(data: Cifar10Data, score_type: str, prune_fraction: float, seed_run: int, *,
                        score_epoch: int = 20, scores_file: Optional[Path] = None, output_root: Path = PRUNING_ROOT,
                        averaged_scores_root: Path = AVERAGED_SCORES_ROOT,
                        total_epochs: int = 200, stop_after_epochs: Optional[int] = None, precision: str = "bf16",
                        mlflow_experiment: Optional[str] = "data-diet-pruning",
                        parent_run_id: Optional[str] = None) -> dict:
    """Train on the pruned set unless an intact finished run already exists. Returns one row of result numbers."""
    if seed_run <= LAST_SEED_RUN_USED_FOR_SCORES and score_type != RANDOM_SCORE_TYPE:
        print(f"  WARNING: seed run {seed_run} is also a score-collection seed; the paper uses different seeds")
    run_directory = run_directory_for(output_root, score_type, prune_fraction, seed_run)
    if run_directory.exists() and not verify_run_directory(run_directory, check_hashes=True):
        print(f"  {run_directory} already complete, skipping")
        return result_row_from_directory(run_directory, score_type, prune_fraction, seed_run, reused_finished_run=True)
    scores = None
    if score_type != RANDOM_SCORE_TYPE:
        scores_file = scores_file or averaged_scores_root / f"epoch_{score_epoch:03d}" / f"{score_type}.npy"
        scores = np.load(scores_file)
        if len(scores) != data.num_train_examples or np.isnan(scores).any():
            raise ValueError(f"{scores_file}: expected {data.num_train_examples} scores without NaN")
    labels = data.train_labels.cpu().numpy()
    kept = select_examples(score_type, scores, prune_fraction, RANDOM_SUBSET_SEED_OFFSET + seed_run, len(labels))
    selection = describe_selection(kept, labels, scores, scores_file, score_type, score_epoch, prune_fraction)
    config = TrainingConfig(
        run_number=seed_run, total_epochs=total_epochs, stop_after_epochs=stop_after_epochs, precision=precision,
        checkpoint_epochs=(), extra_parameters={
            "purpose": "pruning", "score_type": score_type, "score_epoch": score_epoch,
            "requested_prune_fraction": prune_fraction, "selection_rule": "remove lowest scores" if scores is not None
            else "random subset", "cutoff_score_lowest_kept": selection.get("cutoff_score_lowest_kept"),
            "scores_file_sha256": selection.get("scores_file_sha256")})
    name = run_name_for(score_type, prune_fraction, seed_run)
    print(f"\n=== {name}: keep {len(kept)} of {len(labels)} examples, per class {list(selection['kept_per_class'].values())}",
          flush=True)
    run_directory.mkdir(parents=True, exist_ok=True)
    np.save(run_directory / "kept_indices.npy", kept)
    (run_directory / "selection_summary.json").write_text(json.dumps(selection, indent=1))
    tags = {"stage": "pruning", "score_type": score_type, "prune_fraction": str(prune_fraction)}
    if parent_run_id:
        tags["mlflow.parentRunId"] = parent_run_id
    train_with_fixed_steps(data, config, run_directory, train_indices=torch.from_numpy(kept).long(),
                           mlflow_experiment=mlflow_experiment, mlflow_run_name=name, mlflow_tags=tags)
    extend_manifest(run_directory, ["kept_indices.npy", "selection_summary.json"])
    problems = verify_run_directory(run_directory, check_hashes=True)
    if problems:
        raise RuntimeError(f"{name}: run directory has problems: {problems}")
    if mlflow_experiment:
        log_selection_to_mlflow(mlflow_experiment, name, run_directory, selection)
    row = result_row_from_directory(run_directory, score_type, prune_fraction, seed_run, reused_finished_run=False)
    print(f"  {name}: final test accuracy {row['final_test_accuracy']:.4f} in {row['train_minutes']:.1f} min", flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-type", required=True, help="el2n_score, grand_score, forgetting_score, loss or random")
    parser.add_argument("--prune-fraction", type=float, required=True)
    parser.add_argument("--seed-run", type=int, default=101)
    parser.add_argument("--score-epoch", type=int, default=20)
    parser.add_argument("--scores-file", type=Path, default=None, help="override the averaged score file")
    parser.add_argument("--output-root", type=Path, default=PRUNING_ROOT)
    parser.add_argument("--averaged-scores-root", type=Path, default=AVERAGED_SCORES_ROOT)
    parser.add_argument("--total-epochs", type=int, default=200)
    parser.add_argument("--stop-after-epochs", type=int, default=None, help="for smoke tests only")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "tf32", "fp32"])
    parser.add_argument("--mlflow-experiment", default="data-diet-pruning")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    row = run_prune_and_train(
        load_cifar10(), args.score_type, args.prune_fraction, args.seed_run, score_epoch=args.score_epoch,
        scores_file=args.scores_file, output_root=args.output_root, averaged_scores_root=args.averaged_scores_root,
        total_epochs=args.total_epochs,
        stop_after_epochs=args.stop_after_epochs, precision=args.precision,
        mlflow_experiment=None if args.no_mlflow else args.mlflow_experiment)
    print(json.dumps(row, indent=1), f"\nelapsed {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
