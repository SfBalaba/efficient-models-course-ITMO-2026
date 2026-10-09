"""Rebuild results/pruning_sweep/pruning_results.csv from EVERY finished pruning run on disk.

    python collect_pruning_results.py [--log-to-mlflow]

run_pruning_sweep.py writes the table from the runs of its own invocation only, so a later sweep (another score type)
overwrites the rows of earlier ones. This script reads the run directories themselves (exps/pruning/<score type>/
prune_XX/seed_YYY, valid manifest required) and therefore always gives the complete picture. Then run
plot_pruning_results.py to draw all score types together.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient

from prune_and_train import PRUNING_ROOT, result_row_from_directory
from run_logging import MLFLOW_TRACKING_URI, verify_run_directory

RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results" / "pruning_sweep"


def collect_finished_runs(pruning_root: Path) -> pd.DataFrame:
    rows, unfinished = [], []
    for run_directory in sorted(pruning_root.glob("*/prune_*/seed_*")):
        if verify_run_directory(run_directory):
            unfinished.append(str(run_directory))
            continue
        score_type = run_directory.parents[1].name
        prune_fraction = int(run_directory.parent.name.split("_")[1]) / 100
        seed_run = int(run_directory.name.split("_")[1])
        rows.append(result_row_from_directory(run_directory, score_type, prune_fraction, seed_run,
                                              reused_finished_run=True))
    if unfinished:
        print(f"ignored {len(unfinished)} unfinished run directories: {unfinished}")
    return pd.DataFrame(rows).sort_values(["seed_run", "score_type", "prune_fraction"]).reset_index(drop=True)


def log_overview_to_mlflow(table: pd.DataFrame, results_directory: Path, experiment: str) -> str:
    """One mlflow run with a point per prune fraction for every score type (a line chart in the mlflow UI)."""
    client = MlflowClient(MLFLOW_TRACKING_URI)
    existing = client.get_experiment_by_name(experiment)
    experiment_id = existing.experiment_id if existing else client.create_experiment(experiment)
    run_id = client.create_run(experiment_id, tags={"mlflow.runName": "pruning_overview_all_score_types",
                                                    "stage": "pruning_overview"}).info.run_id
    for _, row in table.iterrows():
        client.log_metric(run_id, f"final_test_accuracy__{row['score_type']}__seed{int(row['seed_run'])}",
                          float(row["final_test_accuracy"]), step=round(row["prune_fraction"] * 100))
    for name in ("pruning_results.csv", "pruning_summary_table.csv", "pruning_accuracy_vs_prune_fraction.png",
                 "pruning_verdicts.txt"):
        if (results_directory / name).is_file():
            client.log_artifact(run_id, str(results_directory / name), "pruning_results")
    client.set_terminated(run_id, "FINISHED")
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pruning-root", type=Path, default=PRUNING_ROOT)
    parser.add_argument("--results-directory", type=Path, default=RESULTS_DIRECTORY)
    parser.add_argument("--log-to-mlflow", action="store_true", help="after plotting: attach the table and the figure")
    parser.add_argument("--mlflow-experiment", default="data-diet-pruning")
    args = parser.parse_args()
    table = collect_finished_runs(args.pruning_root)
    args.results_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.results_directory / "pruning_results.csv", index=False)
    summary = table.groupby(["score_type", "seed_run"]).size().rename("finished_runs")
    print(f"{len(table)} finished runs written to {args.results_directory / 'pruning_results.csv'}\n{summary.to_string()}")
    if args.log_to_mlflow:
        print("mlflow overview run:", log_overview_to_mlflow(table, args.results_directory, args.mlflow_experiment))
    return 0


if __name__ == "__main__":
    sys.exit(main())
