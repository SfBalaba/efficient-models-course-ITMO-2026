"""Plot and judge the pruning sweep: python plot_pruning_results.py

Reads results/pruning_sweep/pruning_results.csv (written by run_pruning_sweep.py) and the full-data baseline from
results/score_collection/summary.json (10 independent runs, so its spread is a measured noise level).
Writes the accuracy-versus-prune-fraction figure, a summary table and a verdict on the claims of the paper into
results/pruning_sweep/, and attaches them to the mlflow "sweep_summary" run.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from run_logging import MLFLOW_TRACKING_URI

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
SCORE_COLORS = {"el2n_score": "#c0392b", "grand_score": "#8e44ad", "forgetting_score": "#16a085",
                "loss": "#d68910", "random": "#2c3e50"}
SIGNIFICANCE_Z = 2.0  # two standard errors of a difference of two measurements


def load_baseline(summary_path: Path) -> dict:
    baseline = json.loads(summary_path.read_text())["baseline_full_data"]
    return {"mean": baseline["mean"] * 100, "std": baseline["std"] * 100, "runs": len(baseline["final_test_accuracy_per_run"])}


def summarize(results: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """One row per (score type, prune fraction): mean and spread over seeds, distance to the baseline in noise units."""
    grouped = results.assign(accuracy=results["final_test_accuracy"] * 100).groupby(["score_type", "prune_fraction"])
    table = grouped["accuracy"].agg(["mean", "min", "max", "count"]).reset_index()
    table = table.rename(columns={"mean": "accuracy_mean", "min": "accuracy_min", "max": "accuracy_max",
                                  "count": "seeds"})
    table["difference_to_baseline_pp"] = table["accuracy_mean"] - baseline["mean"]
    standard_error = baseline["std"] * np.sqrt(1 / table["seeds"] + 1 / baseline["runs"])
    table["difference_to_baseline_in_sigmas"] = table["difference_to_baseline_pp"] / standard_error
    return table


def make_figure(table: pd.DataFrame, baseline: dict):
    figure, (main, gap) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    percent = lambda fractions: np.asarray(fractions) * 100
    main.axhspan(baseline["mean"] - 2 * baseline["std"], baseline["mean"] + 2 * baseline["std"], color="grey", alpha=0.18,
                 label=f"full data {baseline['mean']:.2f} ± 2σ (σ={baseline['std']:.2f}, n={baseline['runs']})")
    main.axhline(baseline["mean"], color="grey", linewidth=1)
    for score_type, frame in table.groupby("score_type"):
        frame = frame.sort_values("prune_fraction")
        color = SCORE_COLORS.get(score_type, None)
        main.plot(percent(frame["prune_fraction"]), frame["accuracy_mean"], marker="o", color=color, label=score_type)
        if (frame["seeds"] > 1).any():
            main.fill_between(percent(frame["prune_fraction"]), frame["accuracy_min"], frame["accuracy_max"], color=color,
                              alpha=0.2)
    main.set(xlabel="pruned part of the training set, %", ylabel="final test accuracy, %",
             title="CIFAR-10, ResNet18: accuracy after pruning by importance score")
    main.legend(fontsize=8)
    main.grid(alpha=0.3)
    random = table[table["score_type"] == "random"].set_index("prune_fraction")["accuracy_mean"]
    for score_type, frame in table[table["score_type"] != "random"].groupby("score_type"):
        frame = frame.set_index("prune_fraction").sort_index()
        common = frame.index.intersection(random.index)
        gap.bar(percent(common) + 1.2 * list(table["score_type"].unique()).index(score_type),
                (frame.loc[common, "accuracy_mean"] - random.loc[common]).to_numpy(), width=1.2,
                color=SCORE_COLORS.get(score_type), label=f"{score_type} - random")
    noise = SIGNIFICANCE_Z * baseline["std"] * np.sqrt(2)
    gap.axhspan(-noise, noise, color="grey", alpha=0.18, label=f"single-run noise (±{noise:.2f} pp)")
    gap.set(xlabel="pruned part, %", ylabel="accuracy gain over random pruning, pp", title="Gain over random")
    gap.legend(fontsize=8)
    gap.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def judge_paper_claims(table: pd.DataFrame, baseline: dict, score_type: str = "el2n_score") -> list:
    """Plain-language verdicts; 'sigma' is the measured spread of the 10 full-data baseline runs."""
    scores = table[table["score_type"] == score_type].set_index("prune_fraction")
    random = table[table["score_type"] == "random"].set_index("prune_fraction")
    lines = []
    if 0.5 in scores.index:
        row = scores.loc[0.5]
        verdict = "holds" if row["difference_to_baseline_in_sigmas"] > -SIGNIFICANCE_Z else "NOT confirmed"
        lines.append(f"[{verdict}] claim 1: half of CIFAR-10 can be pruned by {score_type} without loss: "
                     f"{row['accuracy_mean']:.2f}% vs {baseline['mean']:.2f}% full data "
                     f"({row['difference_to_baseline_in_sigmas']:+.1f} sigma)")
    better = [f for f in scores.index if f in random.index and
              scores.loc[f, "accuracy_mean"] - random.loc[f, "accuracy_mean"] > SIGNIFICANCE_Z * baseline["std"] * np.sqrt(
                  1 / scores.loc[f, "seeds"] + 1 / random.loc[f, "seeds"])]
    comparable = [f for f in scores.index if f in random.index]
    lines.append(f"[{'holds' if comparable and len(better) >= len(comparable) // 2 else 'NOT confirmed'}] claim 2: "
                 f"{score_type} beats random pruning of the same size significantly at prune fractions {sorted(better)} "
                 f"(compared: {sorted(comparable)})")
    if 0.7 in scores.index and 0.5 in scores.index:
        drop = scores.loc[0.5, "accuracy_mean"] - scores.loc[0.7, "accuracy_mean"]
        lines.append(f"[{'holds' if drop > 1.0 else 'NOT confirmed'}] claim 3: sharp degradation at heavy pruning: "
                     f"{drop:.2f} pp lost between 50% and 70% pruned")
    return lines


def attach_to_mlflow(results_directory: Path, table: pd.DataFrame, verdicts: list) -> None:
    client = MlflowClient(MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name("data-diet-pruning")
    summaries = client.search_runs([experiment.experiment_id], "tags.stage = 'pruning_sweep'",
                                   order_by=["attributes.start_time DESC"], max_results=1) if experiment else []
    if not summaries:
        print("no sweep_summary mlflow run found, nothing attached")
        return
    run_id = summaries[0].info.run_id
    for _, row in table.iterrows():
        step = round(row["prune_fraction"] * 100)
        client.log_metric(run_id, f"accuracy_mean__{row['score_type']}", float(row["accuracy_mean"]), step=step)
        client.log_metric(run_id, f"difference_to_baseline_sigmas__{row['score_type']}",
                          float(row["difference_to_baseline_in_sigmas"]), step=step)
    client.set_tag(run_id, "verdicts", " | ".join(verdicts)[:5000])
    for name in ("pruning_accuracy_vs_prune_fraction.png", "pruning_summary_table.csv", "pruning_results.csv",
                 "pruning_verdicts.txt"):
        client.log_artifact(run_id, str(results_directory / name), "pruning_results")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-directory", type=Path, default=RESULTS_ROOT / "pruning_sweep")
    parser.add_argument("--baseline-summary", type=Path, default=RESULTS_ROOT / "score_collection" / "summary.json")
    parser.add_argument("--score-type", default="el2n_score", help="score type judged against random")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    results = pd.read_csv(args.results_directory / "pruning_results.csv")
    baseline = load_baseline(args.baseline_summary)
    table = summarize(results, baseline)
    table.to_csv(args.results_directory / "pruning_summary_table.csv", index=False)
    figure = make_figure(table, baseline)
    figure.savefig(args.results_directory / "pruning_accuracy_vs_prune_fraction.png", dpi=110)
    verdicts = judge_paper_claims(table, baseline, args.score_type)
    (args.results_directory / "pruning_verdicts.txt").write_text("\n".join(verdicts) + "\n")
    print(table.round(3).to_string(index=False))
    print("\n".join(verdicts))
    if not args.no_mlflow:
        attach_to_mlflow(args.results_directory, table, verdicts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
