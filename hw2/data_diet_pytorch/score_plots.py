"""Figures for the collected scores. Pure functions: arrays in, matplotlib figure out (no file or mlflow access)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURE_DPI = 110


def plot_score_histograms(scores_by_epoch: dict, title: str, bins: int = 60):
    """One histogram per epoch (log-scaled counts): how the score distribution moves during training."""
    columns = min(4, len(scores_by_epoch))
    rows = int(np.ceil(len(scores_by_epoch) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 2.6 * rows), squeeze=False)
    for axis, (epoch, values) in zip(axes.flat, scores_by_epoch.items()):
        finite = values[np.isfinite(values)]
        axis.hist(finite, bins=bins, color="#3b6ea5")
        axis.set_yscale("log")
        axis.set_title(f"epoch {epoch}", fontsize=9)
    for axis in list(axes.flat)[len(scores_by_epoch):]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_quantiles_over_epochs(scores_by_epoch: dict, title: str):
    epochs = sorted(scores_by_epoch)
    figure, axis = plt.subplots(figsize=(6, 3.8))
    for quantile, style in ((10, ":"), (50, "-"), (90, "--"), (99, "-.")):
        axis.plot(epochs, [np.percentile(scores_by_epoch[e][np.isfinite(scores_by_epoch[e])], quantile) for e in epochs],
                  style, marker="o", markersize=3, label=f"p{quantile}")
    axis.set(xlabel="epoch", ylabel="score", title=title)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_reliability(number_of_runs: list, curves: dict, title: str):
    """Spearman correlation between the mean over k runs and the mean over all runs (cf. figures 10-11 of the paper)."""
    figure, axis = plt.subplots(figsize=(6, 3.8))
    for name, (mean, std) in curves.items():
        mean, std = np.asarray(mean), np.asarray(std)
        axis.plot(number_of_runs, mean, marker="o", label=name)
        axis.fill_between(number_of_runs, mean - std, mean + std, alpha=0.2)
    axis.set(xlabel="number of runs averaged", ylabel="Spearman with the all-runs average", title=title, ylim=(None, 1.0))
    axis.legend()
    figure.tight_layout()
    return figure


def plot_correlation_heatmap(matrix: np.ndarray, labels: list, title: str):
    figure, axis = plt.subplots(figsize=(0.55 * len(labels) + 3, 0.5 * len(labels) + 2.5))
    image = axis.imshow(matrix, vmin=min(0.0, float(np.nanmin(matrix))), vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=7, color="white")
    axis.set_title(title, fontsize=10)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    return figure


def plot_score_scatter(x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, title: str):
    figure, axis = plt.subplots(figsize=(5, 4.2))
    mask = np.isfinite(x) & np.isfinite(y)
    hexes = axis.hexbin(x[mask], y[mask], gridsize=70, bins="log", cmap="magma")
    axis.set(xlabel=x_label, ylabel=y_label, title=title)
    figure.colorbar(hexes, ax=axis, label="log10(count)")
    figure.tight_layout()
    return figure


def plot_class_composition(counts_by_prune_fraction: dict, class_names: tuple, title: str):
    """Share of every class in the kept subset for each prune fraction (hard classes dominate after pruning)."""
    fractions = sorted(counts_by_prune_fraction)
    shares = np.array([counts_by_prune_fraction[f] / counts_by_prune_fraction[f].sum() for f in fractions])
    figure, axis = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(fractions))
    for index, name in enumerate(class_names):
        axis.bar([f"{f:.1f}" for f in fractions], shares[:, index], bottom=bottom, label=name)
        bottom += shares[:, index]
    axis.axhline(0.1, color="black", linewidth=0.5, linestyle=":")
    axis.set(xlabel="prune fraction", ylabel="share of the kept examples", title=title)
    axis.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    return figure


def plot_extreme_examples(images_uint8: np.ndarray, labels: np.ndarray, scores: np.ndarray, class_names: tuple,
                          per_class: int = 8):
    """Per class: lowest-score (easy) and highest-score (hard) examples, like figures 7-8 of the paper."""
    figure, axes = plt.subplots(len(class_names), 2 * per_class, figsize=(1.0 * 2 * per_class, 1.05 * len(class_names)))
    for row, name in enumerate(class_names):
        indices = np.where(labels == row)[0]
        ordered = indices[np.argsort(scores[indices])]
        chosen = list(ordered[:per_class]) + list(ordered[-per_class:])
        for column, index in enumerate(chosen):
            axes[row, column].imshow(images_uint8[index])
            axes[row, column].axis("off")
        axes[row, 0].set_title(name, fontsize=7, loc="left")
    figure.suptitle(f"left of each row: {per_class} lowest-score (easy) | right: {per_class} highest-score (hard)", fontsize=9)
    figure.tight_layout()
    return figure


def plot_accuracy_curves(accuracy_by_run: dict, title: str):
    """Test accuracy per epoch for every run plus mean and the spread of the final accuracy."""
    figure, (curve_axis, final_axis) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 1]})
    stacked = np.stack(list(accuracy_by_run.values()))
    for curve in stacked:
        curve_axis.plot(curve, linewidth=0.6, alpha=0.5)
    curve_axis.plot(stacked.mean(axis=0), color="black", linewidth=1.5, label="mean")
    curve_axis.set(xlabel="epoch", ylabel="test accuracy", ylim=(0.7, 1.0), title=title)
    curve_axis.legend()
    final_axis.scatter(np.zeros(len(stacked)), stacked[:, -1] * 100)
    final_axis.set(ylabel="final test accuracy, %", xticks=[], title=f"{stacked[:, -1].mean() * 100:.2f} ± "
                   f"{stacked[:, -1].std(ddof=1) * 100 if len(stacked) > 1 else 0:.2f}")
    figure.tight_layout()
    return figure
