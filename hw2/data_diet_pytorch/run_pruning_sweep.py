"""Run the pruning sweep over prune fractions and score types: python run_pruning_sweep.py [options]

Runs are ordered by priority (fractions 0.5, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1; every fraction trains the score-based subset
first, then the random control), so stopping early still leaves a readable curve. Finished runs are skipped, so the
command can simply be started again after an interruption. Results are written to results/pruning_sweep/ after
every run, and mlflow gets a parent run "sweep_summary" with one point per prune fraction for each score type.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient

from cifar10_data import load_cifar10
from prune_and_train import AVERAGED_SCORES_ROOT, PRUNING_ROOT, run_directory_for, run_prune_and_train
from run_logging import MLFLOW_TRACKING_URI, verify_run_directory

PRIORITY_FRACTIONS = (0.5, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1)
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results" / "pruning_sweep"
RANDOM_SCORE_TYPE = "random"


def start_sweep_summary_run(experiment: str, seed_runs: list) -> tuple:
    client = MlflowClient(MLFLOW_TRACKING_URI)
    experiment_object = client.get_experiment_by_name(experiment)
    experiment_id = experiment_object.experiment_id if experiment_object else client.create_experiment(experiment)
    run = client.create_run(experiment_id, tags={"mlflow.runName": f"sweep_summary_seeds_{'_'.join(map(str, seed_runs))}",
                                                  "stage": "pruning_sweep"})
    return client, run.info.run_id


def write_results(rows: list, results_directory: Path) -> pd.DataFrame:
    results_directory.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values(["seed_run", "score_type", "prune_fraction"])
    table.to_csv(results_directory / "pruning_results.csv", index=False)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fractions", type=float, nargs="+", default=list(PRIORITY_FRACTIONS))
    parser.add_argument("--score-types", nargs="+", default=["el2n_score", RANDOM_SCORE_TYPE])
    parser.add_argument("--seed-runs", type=int, nargs="+", default=[101])
    parser.add_argument("--score-epoch", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=PRUNING_ROOT)
    parser.add_argument("--averaged-scores-root", type=Path, default=AVERAGED_SCORES_ROOT)
    parser.add_argument("--results-directory", type=Path, default=RESULTS_DIRECTORY)
    parser.add_argument("--total-epochs", type=int, default=200)
    parser.add_argument("--stop-after-epochs", type=int, default=None, help="for smoke tests only")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "tf32", "fp32"])
    parser.add_argument("--mlflow-experiment", default="data-diet-pruning")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--time-budget-hours", type=float, default=None, help="do not start a run that would not fit")
    parser.add_argument("--minutes-per-run", type=float, default=13.5, help="used for the ETA and the time budget")
    parser.add_argument("--accuracy-gate", type=float, default=0.945,
                        help="stop if the first score-based run with prune fraction <= 0.5 ends below this accuracy")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = [(seed, fraction, score_type) for seed in args.seed_runs for fraction in args.fractions
            for score_type in args.score_types]
    pending = [item for item in plan if not (
        (directory := run_directory_for(args.output_root, item[2], item[1], item[0])).exists()
        and not verify_run_directory(directory))]
    print(f"{len(plan)} runs planned, {len(plan) - len(pending)} already complete, {len(pending)} to train "
          f"(~{len(pending) * args.minutes_per_run / 60:.1f} h at {args.minutes_per_run} min per run)")
    for seed, fraction, score_type in plan:
        mark = "todo" if (seed, fraction, score_type) in pending else "done"
        print(f"  [{mark}] seed {seed} prune {fraction:.1f} {score_type}")
    if args.dry_run:
        return 0

    data = load_cifar10()
    client, parent_run_id = (None, None) if args.no_mlflow else start_sweep_summary_run(args.mlflow_experiment,
                                                                                         args.seed_runs)
    experiment = None if args.no_mlflow else args.mlflow_experiment
    rows, started, gate_checked, status = [], time.perf_counter(), False, "FINISHED"
    try:
        for seed, fraction, score_type in plan:
            will_train = (seed, fraction, score_type) in pending
            elapsed_hours = (time.perf_counter() - started) / 3600
            if will_train and args.time_budget_hours and elapsed_hours + args.minutes_per_run / 60 > args.time_budget_hours:
                print(f"time budget of {args.time_budget_hours} h reached after {elapsed_hours:.2f} h, stopping")
                break
            row = run_prune_and_train(
                data, score_type, fraction, seed, score_epoch=args.score_epoch, output_root=args.output_root,
                averaged_scores_root=args.averaged_scores_root,
                total_epochs=args.total_epochs, stop_after_epochs=args.stop_after_epochs, precision=args.precision,
                mlflow_experiment=experiment, parent_run_id=parent_run_id)
            rows.append(row)
            table = write_results(rows, args.results_directory)
            if client:
                client.log_metric(parent_run_id, f"final_test_accuracy__{score_type}__seed{seed}",
                                  row["final_test_accuracy"], step=round(fraction * 100))
            if not gate_checked and score_type != RANDOM_SCORE_TYPE and fraction <= 0.5 and will_train:
                gate_checked = True
                if row["final_test_accuracy"] < args.accuracy_gate:
                    print(f"\nGATE FAILED: {score_type} at prune fraction {fraction} reached "
                          f"{row['final_test_accuracy']:.4f} < {args.accuracy_gate}. Stopping to investigate "
                          f"(score quality, seeds, BatchNorm) before spending more GPU time.")
                    status = "KILLED"
                    return 3
    finally:
        if client:
            client.set_terminated(parent_run_id, status)
    if rows:
        print("\n=== final test accuracy (%), rows = prune fraction")
        print((table.pivot_table(index="prune_fraction", columns=["score_type", "seed_run"],
                                 values="final_test_accuracy") * 100).round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
