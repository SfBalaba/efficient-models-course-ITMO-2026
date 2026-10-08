"""Combine the score-collection runs: python aggregate_scores.py [--runs 1 2 ...] [--per-run-figures]

Reads exps/score_collection/run_XX/scores and writes
    exps/score_collection/aggregated/epoch_XXX/<score>.npy   averages over runs (raw, not committed)
    results/score_collection/                                small tables, figures, epoch-20 averages (committed)
and logs everything to an mlflow run called "aggregation".
The averaged scores are exactly what the pruning step consumes: mean over runs, +inf of forgetting is kept as in the
original get_mean_score.py (examples never learned in some run are treated as the most important).
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from scipy.stats import rankdata
from torchvision.datasets import CIFAR10

import score_plots
from cifar10_data import CLASS_NAMES, DEFAULT_DATA_ROOT
from run_logging import MLFLOW_TRACKING_URI, verify_run_directory

COLLECTION_ROOT = Path(__file__).resolve().parent / "exps" / "score_collection"
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results" / "score_collection"
EXCLUDED_FROM_AVERAGING = {"predicted_class", "first_learned_step"}
PRUNE_FRACTIONS = tuple(round(0.1 * step, 1) for step in range(1, 8))
RELIABILITY_DRAWS = 20


def rank_safe(values: np.ndarray) -> np.ndarray:
    """Forgetting uses +inf for never-learned examples; for correlations they get the largest rank."""
    finite = np.isfinite(values)
    return np.where(finite, values, (values[finite].max() if finite.any() else 0) + 1)


def to_ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(rank_safe(values))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(to_ranks(a), to_ranks(b))[0, 1])


def jaccard_of_top_half(a: np.ndarray, b: np.ndarray) -> float:
    top_a, top_b = (set(np.argsort(rank_safe(v), kind="stable")[len(v) // 2:]) for v in (a, b))
    return len(top_a & top_b) / len(top_a | top_b)


def score_epochs_of(run_directory: Path) -> list:
    return sorted(int(folder.name.split("_")[1]) for folder in (run_directory / "scores").glob("epoch_*"))


def load_scores(run_directories: list, epoch: int, name: str) -> list:
    return [np.load(path) for path in (run / "scores" / f"epoch_{epoch:03d}" / f"{name}.npy" for run in run_directories)
            if path.is_file()]


def reliability_curve(per_run_values: list, generator: np.random.Generator) -> tuple:
    """Spearman between the average of k random runs and the average of all runs, for k = 1 .. N-1."""
    stacked = np.stack([rank_safe(values) for values in per_run_values])
    reference = to_ranks(stacked.mean(axis=0))
    means, stds = [], []
    for k in range(1, len(stacked)):
        correlations = [np.corrcoef(rankdata(stacked[generator.choice(len(stacked), k, replace=False)].mean(axis=0)),
                                    reference)[0, 1] for _ in range(RELIABILITY_DRAWS)]
        means.append(float(np.mean(correlations)))
        stds.append(float(np.std(correlations)))
    return list(range(1, len(stacked))), means, stds


def save_figure(figure, name: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    figure.savefig(path, dpi=score_plots.FIGURE_DPI)
    score_plots.plt.close(figure)
    return path


def make_run_figures(run_directory: Path, directory: Path) -> list:
    """Figures of a single run (histograms per epoch, quantiles, EL2N vs GraNd, forgetting at the end)."""
    epochs = score_epochs_of(run_directory)
    early = [epoch for epoch in epochs if epoch <= 20]
    el2n = {epoch: np.load(run_directory / "scores" / f"epoch_{epoch:03d}" / "el2n_score.npy") for epoch in epochs}
    paths = [save_figure(score_plots.plot_score_histograms({e: el2n[e] for e in early}, "EL2N, one run"),
                         "el2n_histograms_early_epochs", directory),
             save_figure(score_plots.plot_quantiles_over_epochs(el2n, "EL2N quantiles, one run"),
                         "el2n_quantiles_over_epochs", directory)]
    reference = max(early)
    grand_file = run_directory / "scores" / f"epoch_{reference:03d}" / "grand_score.npy"
    if grand_file.is_file():
        paths.append(save_figure(score_plots.plot_score_scatter(el2n[reference], np.load(grand_file), "EL2N", "GraNd",
                                                                f"epoch {reference}, one run"), "el2n_vs_grand", directory))
    forgetting = np.load(run_directory / "scores" / f"epoch_{epochs[-1]:03d}" / "forgetting_score.npy")
    paths.append(save_figure(score_plots.plot_score_histograms(
        {f"{epochs[-1]} (never learned: {np.isinf(forgetting).sum()})": forgetting}, "forgetting score, one run"),
        "forgetting_histogram", directory))
    return paths


def log_per_run_figures(run_directories: list, experiment: str) -> None:
    client = MlflowClient(MLFLOW_TRACKING_URI)
    experiment_id = client.get_experiment_by_name(experiment).experiment_id
    for run_directory in run_directories:
        number = int(run_directory.name.split("_")[1])
        figures_directory = run_directory / "figures"
        paths = make_run_figures(run_directory, figures_directory)
        found = client.search_runs([experiment_id], f"tags.mlflow.runName = 'score_run_{number:02d}'",
                                   order_by=["attributes.start_time DESC"], max_results=1)
        for path in paths:
            if found:
                client.log_artifact(found[0].info.run_id, str(path), "figures")
        print(f"  figures of {run_directory.name}: {[p.name for p in paths]}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=COLLECTION_ROOT)
    parser.add_argument("--runs", type=int, nargs="+", default=None)
    parser.add_argument("--per-run-figures", action="store_true")
    parser.add_argument("--mlflow-experiment", default="data-diet-score-collection")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--results-directory", type=Path, default=RESULTS_DIRECTORY)
    args = parser.parse_args()

    candidates = sorted(args.root.glob("run_[0-9][0-9]"))
    run_directories = [d for d in candidates if (args.runs is None or int(d.name[4:]) in args.runs)
                       and not verify_run_directory(d)]
    skipped = [d.name for d in candidates if d not in run_directories]
    print(f"aggregating {len(run_directories)} complete runs: {[d.name for d in run_directories]}; skipped: {skipped}")
    if len(run_directories) < 2:
        print("need at least two complete runs")
        return 2
    results, aggregated_root = args.results_directory, args.root / "aggregated"
    results.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(0)
    labels = np.load(run_directories[0] / "scores" / "train_labels.npy").astype(int)
    epochs = score_epochs_of(run_directories[0])
    metrics, tables = {}, {}
    summary = {"runs_used": [d.name for d in run_directories], "runs_skipped": skipped, "score_epochs": epochs,
               "runs_averaged_per_score": {}}

    # 1. averages over runs
    averaged = {}
    for epoch in epochs:
        names = sorted({file.stem for run in run_directories
                        for file in (run / "scores" / f"epoch_{epoch:03d}").glob("*.npy")} - EXCLUDED_FROM_AVERAGING)
        for name in names:
            arrays = load_scores(run_directories, epoch, name)
            averaged[(name, epoch)] = np.mean(np.stack(arrays), axis=0)
            summary["runs_averaged_per_score"][f"{name}@{epoch}"] = len(arrays)
            target = aggregated_root / f"epoch_{epoch:03d}"
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / f"{name}.npy", averaged[(name, epoch)].astype(np.float32))
    reference_epoch = max(epoch for epoch in epochs if epoch <= 20)
    last_epoch = epochs[-1]

    # 2. reliability of the average as a function of the number of runs
    curves = {}
    reliability_targets = [("el2n_score", reference_epoch), ("grand_score", reference_epoch), ("loss", reference_epoch),
                           ("forgetting_score", last_epoch)] + ([("el2n_score", 10)] if 10 in epochs else [])
    for name, epoch in reliability_targets:
        arrays = load_scores(run_directories, epoch, name)
        if len(arrays) >= 3:
            counts, means, stds = reliability_curve(arrays, generator)
            curves[f"{name}@{epoch} ({len(arrays)} runs)"] = (means, stds)
            metrics.update({f"reliability_{name}_epoch{epoch}_k{k}": m for k, m in zip(counts, means)})
            tables[f"reliability_{name}_epoch{epoch}"] = pd.DataFrame({"runs_averaged": counts, "spearman_mean": means,
                                                                      "spearman_std": stds})
    figures = []
    if curves:
        longest = max(len(mean) for mean, _ in curves.values())
        figures.append(("reliability_vs_number_of_runs", score_plots.plot_reliability(
            list(range(1, longest + 1)), {k: (m + [np.nan] * (longest - len(m)), s + [np.nan] * (longest - len(s)))
                                          for k, (m, s) in curves.items()}, "How many runs are needed")))

    # 3. agreement between individual runs
    for name, epoch in (("el2n_score", reference_epoch), ("grand_score", reference_epoch), ("forgetting_score", last_epoch)):
        arrays = load_scores(run_directories, epoch, name)
        if len(arrays) >= 2:
            ranks = np.stack([to_ranks(values) for values in arrays])
            matrix = np.corrcoef(ranks)
            metrics[f"inter_run_spearman_{name}_epoch{epoch}"] = float(matrix[np.triu_indices(len(matrix), 1)].mean())
            figures.append((f"inter_run_spearman_{name}_epoch{epoch}", score_plots.plot_correlation_heatmap(
                matrix, [str(i + 1) for i in range(len(arrays))], f"{name} at epoch {epoch}: Spearman between runs")))

    # 4. how early the ranking is already the final one (averaged over runs)
    for name in ("el2n_score", "loss"):
        matrix = np.array([[spearman(averaged[(name, a)], averaged[(name, b)]) for b in epochs] for a in epochs])
        figures.append((f"{name}_rank_agreement_between_epochs", score_plots.plot_correlation_heatmap(
            matrix, [str(e) for e in epochs], f"{name}: Spearman between epochs (averaged over runs)")))
        metrics.update({f"{name}_spearman_epoch{e}_vs_epoch{reference_epoch}": spearman(averaged[(name, e)],
                        averaged[(name, reference_epoch)]) for e in epochs})
        tables[f"{name}_rank_agreement_between_epochs"] = pd.DataFrame(matrix, index=epochs, columns=epochs)

    # 5. agreement between score types
    chosen = [(name, epoch) for name, epoch in (("el2n_score", reference_epoch), ("grand_score", reference_epoch),
                                                ("loss", reference_epoch), ("forgetting_score", reference_epoch),
                                                ("forgetting_score", last_epoch)) if (name, epoch) in averaged]
    labels_of_scores = [f"{name}_epoch{epoch}" for name, epoch in chosen]
    for title, function, key in (("Spearman between score types", spearman, "spearman"),
                                 ("overlap (Jaccard) of the hardest halves", jaccard_of_top_half, "jaccard_top_half")):
        matrix = np.array([[function(averaged[a], averaged[b]) for b in chosen] for a in chosen])
        figures.append((f"score_types_{key}", score_plots.plot_correlation_heatmap(matrix, labels_of_scores, title)))
        tables[f"score_types_{key}"] = pd.DataFrame(matrix, index=labels_of_scores, columns=labels_of_scores)
        metrics.update({f"{key}_{a}_vs_{b}": matrix[i, j] for i, a in enumerate(labels_of_scores)
                        for j, b in enumerate(labels_of_scores) if i < j})
    if ("grand_score", reference_epoch) in averaged:
        figures.append(("el2n_vs_grand_averaged", score_plots.plot_score_scatter(
            averaged[("el2n_score", reference_epoch)], averaged[("grand_score", reference_epoch)], "EL2N", "GraNd",
            f"averaged over runs, epoch {reference_epoch}")))

    # 6. which examples would be kept: class composition after pruning by the averaged EL2N
    el2n_reference = averaged[("el2n_score", reference_epoch)]
    by_importance = np.argsort(el2n_reference, kind="stable")
    counts_by_fraction, rows = {}, []
    for fraction in PRUNE_FRACTIONS:
        kept = by_importance[int(round(fraction * len(by_importance))):]
        counts_by_fraction[fraction] = np.bincount(labels[kept], minlength=len(CLASS_NAMES))
        rows.append({"prune_fraction": fraction, "kept_examples": len(kept),
                     **{name: int(count) for name, count in zip(CLASS_NAMES, counts_by_fraction[fraction])}})
    tables[f"kept_class_counts_el2n_epoch{reference_epoch}"] = pd.DataFrame(rows)
    figures.append(("kept_class_composition", score_plots.plot_class_composition(
        counts_by_fraction, CLASS_NAMES, f"Classes kept after pruning by EL2N (epoch {reference_epoch})")))

    # 7. baseline accuracy on all data: every run is an independent full-data training
    curves_by_run = {}
    for run in run_directories:
        table = pd.read_csv(run / "epoch_metrics.csv")
        curves_by_run[run.name] = table["test_accuracy"].to_numpy()
    shortest = min(len(curve) for curve in curves_by_run.values())
    curves_by_run = {name: curve[:shortest] for name, curve in curves_by_run.items()}
    finals = np.array([curve[-1] for curve in curves_by_run.values()])
    summary["baseline_full_data"] = {"epochs": shortest - 1, "final_test_accuracy_per_run": finals.tolist(),
                                     "mean": float(finals.mean()), "std": float(finals.std(ddof=1)),
                                     "min": float(finals.min()), "max": float(finals.max())}
    metrics.update(baseline_final_accuracy_mean=finals.mean(), baseline_final_accuracy_std=finals.std(ddof=1),
                   baseline_final_accuracy_min=finals.min(), baseline_final_accuracy_max=finals.max())
    tables["baseline_final_accuracy_per_run"] = pd.DataFrame({"run": list(curves_by_run), "final_test_accuracy": finals})
    figures.append(("baseline_accuracy_curves", score_plots.plot_accuracy_curves(curves_by_run, "Full-data baseline runs")))

    # 8. distributions and extreme examples
    early = [epoch for epoch in epochs if epoch <= 20]
    figures.append(("el2n_histograms_early_epochs", score_plots.plot_score_histograms(
        {e: averaged[("el2n_score", e)] for e in early}, "EL2N averaged over runs")))
    figures.append(("el2n_quantiles_over_epochs", score_plots.plot_quantiles_over_epochs(
        {e: averaged[("el2n_score", e)] for e in epochs}, "EL2N quantiles (averaged over runs)")))
    images = CIFAR10(DEFAULT_DATA_ROOT, train=True, download=False).data
    figures.append(("easiest_and_hardest_examples", score_plots.plot_extreme_examples(
        images, labels, el2n_reference, CLASS_NAMES)))

    # write everything
    for name, figure in figures:
        save_figure(figure, name, results)
    for name, table in tables.items():
        table.to_csv(results / f"{name}.csv", index=not name.startswith(("kept", "baseline", "reliability")))
    for name, epoch in (("el2n_score", reference_epoch), ("grand_score", reference_epoch), ("forgetting_score", last_epoch)):
        if (name, epoch) in averaged:
            np.save(results / f"{name}_epoch{epoch:03d}_mean_over_runs.npy", averaged[(name, epoch)].astype(np.float32))
    summary["metrics"] = {key: float(value) for key, value in metrics.items()}
    (results / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {len(figures)} figures, {len(tables)} tables, summary.json to {results}")
    print(f"baseline final test accuracy: {finals.mean() * 100:.2f} ± {finals.std(ddof=1) * 100:.2f} (n={len(finals)})")

    if not args.no_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name="aggregation", tags={"stage": "score_aggregation"}):
            mlflow.log_params({"runs_used": json.dumps(summary["runs_used"]), "num_runs": len(run_directories),
                               "reference_epoch": reference_epoch, "last_epoch": last_epoch})
            mlflow.log_metrics({key: float(value) for key, value in metrics.items() if np.isfinite(value)})
            mlflow.log_artifacts(str(results), "aggregated_results")
        if args.per_run_figures:
            log_per_run_figures(run_directories, args.mlflow_experiment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
