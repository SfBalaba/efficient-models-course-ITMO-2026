"""Графики по results/measurements.csv (только измерения, без предсказаний).

Запуск:
    python plot_measurements.py
    python plot_measurements.py --csv results/measurements.csv --out results/figures

Закрашенные маркеры — базовая сетка, полые — валидационные точки
(случайные S или B). Пунктирные линии — случайные S / B.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HW_DIR = Path(__file__).resolve().parent

# порядковая шкала одного оттенка (светлый -> тёмный), шаги 250..700 синей рампы
ORDINAL_STEPS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                 "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"  # опорные линии (лимит памяти, idle power)
OOM_FILL = "#eeede9"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 1.6,
    "lines.markersize": 5.5,
    "figure.dpi": 110,
    "savefig.dpi": 160,
})


def ordinal_colors(n):
    cmap = LinearSegmentedColormap.from_list("ordinal", ORDINAL_STEPS)
    return [cmap(t) for t in np.linspace(0, 1, n)] if n > 1 else [cmap(0.6)]


SEQ_CMAP = LinearSegmentedColormap.from_list("seq", SEQUENTIAL_STEPS)


def load(csv_path):
    df = pd.read_csv(csv_path)
    df["is_validation"] = df["is_validation"].astype(bool)
    return df


def load_env(csv_path):
    env_path = csv_path.with_name("env.json")
    return json.loads(env_path.read_text()) if env_path.exists() else {}


def style_legend(ax, group_name, extra=()):
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], color=INK_2, marker="o", linestyle="none", label="базовая сетка"),
        Line2D([], [], color=INK_2, marker="o", markerfacecolor=SURFACE, linestyle="none", label="валидация"),
        *extra,
    ]
    ax.legend(handles=handles, title=group_name, title_fontsize=9, loc="center left",
              bbox_to_anchor=(1.01, 0.5))


def plot_lines(ax, df, x, y, group, random_groups=(), scale_y=1.0):
    """Линия на каждое значение group; валидационные точки — полые маркеры."""
    values = sorted(df[group].unique())
    for color, val in zip(ordinal_colors(len(values)), values):
        sub = df[df[group] == val].sort_values(x)
        ls = "--" if val in random_groups else "-"
        ax.plot(sub[x], sub[y] * scale_y, color=color, linestyle=ls, label=f"{val}", zorder=2)
        base, val_pts = sub[~sub["is_validation"]], sub[sub["is_validation"]]
        ax.plot(base[x], base[y] * scale_y, "o", color=color, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.plot(val_pts[x], val_pts[y] * scale_y, "o", color=color, markerfacecolor=SURFACE,
                markeredgewidth=1.4, zorder=3)


def log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def save(fig, out_dir, name):
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# --------------------------------------------------------------------------- figures

def fig_latency_vs_batch(df, env, out):
    ok = df[df.status == "OK"]
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lines(ax, ok, "B", "latency_s", "S", env.get("random_S", ()), scale_y=1e3)
    log2_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel("latency, мс (медиана, log)")
    ax.set_title("Latency одного forward pass vs batch size")
    style_legend(ax, "S, px")
    save(fig, out, "latency_vs_batch.png")


def fig_latency_vs_image_size(df, env, out):
    ok = df[df.status == "OK"]
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lines(ax, ok, "S", "latency_s", "B", env.get("random_B", ()), scale_y=1e3)
    ax.set_yscale("log")
    ax.set_xlabel("image size S, px")
    ax.set_ylabel("latency, мс (медиана, log)")
    ax.set_title("Latency одного forward pass vs image size")
    style_legend(ax, "B")
    save(fig, out, "latency_vs_image_size.png")


def fig_latency_vs_pixels(df, env, out):
    """Все точки на одной оси: объём входа B·S² — видно плато launch-bound и линейный участок."""
    ok = df[df.status == "OK"].assign(pixels=lambda d: d.B * d.S ** 2)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lines(ax, ok, "pixels", "latency_s", "S", env.get("random_S", ()), scale_y=1e3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("B · S², пикселей на входе (log)")
    ax.set_ylabel("latency, мс (медиана, log)")
    ax.set_title("Latency vs объём входа")
    style_legend(ax, "S, px")
    save(fig, out, "latency_vs_pixels.png")


def fig_latency_spread(df, env, out):
    """Разброс замеров: p10–p90 относительно медианы."""
    ok = df[df.status == "OK"].assign(spread=lambda d: (d.latency_p90_s - d.latency_p10_s) / d.latency_s * 100)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lines(ax, ok, "B", "spread", "S", env.get("random_S", ()))
    log2_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xlabel("batch size B")
    ax.set_ylabel("(p90 − p10) / медиана, %")
    ax.set_title("Стабильность замеров latency")
    style_legend(ax, "S, px")
    save(fig, out, "latency_spread.png")


def fig_throughput(df, env, out):
    ok = df[df.status == "OK"].assign(throughput=lambda d: d.B / d.latency_s)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lines(ax, ok, "B", "throughput", "S", env.get("random_S", ()))
    log2_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel("throughput, изображений/с (log)")
    ax.set_title("Пропускная способность")
    style_legend(ax, "S, px")
    save(fig, out, "throughput_vs_batch.png")


def fig_memory(df, env, out):
    ok = df[df.status == "OK"]
    oom = df[df.status == "OOM"]
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lines(ax, ok, "B", "memory_bytes", "S", env.get("random_S", ()), scale_y=1 / 2**20)
    extra = []
    total = env.get("gpu_total_memory_bytes")
    if total:
        ax.axhline(total / 2**20, color=REFERENCE, linestyle=":", linewidth=1.2)
        ax.annotate(f"память GPU {total / 2**30:.1f} GiB", (1, total / 2**20), xytext=(2, 4),
                    textcoords="offset points", color=INK_2, fontsize=8.5)
    if len(oom) and total:
        colors = dict(zip(sorted(df.S.unique()), ordinal_colors(df.S.nunique())))
        for _, r in oom.iterrows():
            ax.plot(r.B, total / 2**20, marker="x", color=colors[r.S], markersize=8, markeredgewidth=2, zorder=4)
        extra = [Line2D([], [], color=INK_2, marker="x", linestyle="none", markeredgewidth=2, label="OOM")]
    log2_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel("max_memory_allocated, MiB (log)")
    ax.set_title("Пиковая память forward pass")
    style_legend(ax, "S, px", extra)
    save(fig, out, "memory_vs_batch.png")


def fig_energy(df, env, out):
    ok = df[(df.status == "OK") & df.energy_j.notna()].assign(energy_per_img=lambda d: d.energy_j / d.B * 1e3)
    if ok.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_lines(ax1, ok, "B", "energy_j", "S", env.get("random_S", ()))
    ax1.set_ylabel("энергия на forward pass, Дж (log)")
    ax1.set_title("Энергия одного прохода")
    plot_lines(ax2, ok, "B", "energy_per_img", "S", env.get("random_S", ()))
    ax2.set_ylabel("энергия на изображение, мДж (log)")
    ax2.set_title("Энергия на одно изображение")
    for ax in (ax1, ax2):
        log2_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_yscale("log")
        ax.set_xlabel("batch size B")
    style_legend(ax2, "S, px")
    save(fig, out, "energy_vs_batch.png")


def fig_power(df, env, out):
    ok = df[(df.status == "OK") & df.avg_power_w.notna()]
    if ok.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lines(ax, ok, "B", "avg_power_w", "S", env.get("random_S", ()))
    for key, label in (("idle_power_w", "idle"), ("power_limit_w", "power limit")):
        if env.get(key):
            ax.axhline(env[key], color=REFERENCE, linestyle=":", linewidth=1.2)
            ax.annotate(f"{label} {env[key]:.0f} W", (1, env[key]), xytext=(2, 4),
                        textcoords="offset points", color=INK_2, fontsize=8.5)
    log2_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xlabel("batch size B")
    ax.set_ylabel("средняя мощность GPU, Вт")
    ax.set_title("Мощность во время непрерывных проходов")
    style_legend(ax, "S, px")
    save(fig, out, "power_vs_batch.png")


def fig_heatmap(df, env, out, column, scale, unit, title, name, fmt):
    s_vals, b_vals = sorted(df.S.unique()), sorted(df.B.unique())
    grid = np.full((len(s_vals), len(b_vals)), np.nan)
    status = np.full(grid.shape, "", dtype=object)
    for _, r in df.iterrows():
        i, j = s_vals.index(r.S), b_vals.index(r.B)
        status[i, j] = r.status
        if r.status == "OK" and pd.notna(r[column]):
            grid[i, j] = r[column] * scale

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.grid(False)
    ax.set_facecolor(OOM_FILL)
    norm = LogNorm(vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    im = ax.imshow(grid, cmap=SEQ_CMAP, norm=norm, aspect="auto", origin="lower")
    for i in range(len(s_vals)):
        for j in range(len(b_vals)):
            if status[i, j] != "OK":
                ax.text(j, i, status[i, j], ha="center", va="center", fontsize=8, color=INK_2, weight="bold")
            elif not np.isnan(grid[i, j]):
                dark = norm(grid[i, j]) > 0.55
                ax.text(j, i, fmt(grid[i, j]), ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if dark else INK)
    rand_s, rand_b = set(env.get("random_S", ())), set(env.get("random_B", ()))
    ax.set_xticks(range(len(b_vals)), [f"{b}*" if b in rand_b else str(b) for b in b_vals])
    ax.set_yticks(range(len(s_vals)), [f"{s}*" if s in rand_s else str(s) for s in s_vals])
    ax.set_xlabel("batch size B   (* — валидационные значения)")
    ax.set_ylabel("image size S, px")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(unit, color=INK_2)
    cbar.outline.set_visible(False)
    save(fig, out, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=HW_DIR / "results" / "measurements.csv")
    p.add_argument("--out", type=Path, default=HW_DIR / "results" / "figures")
    args = p.parse_args()

    df = load(args.csv)
    env = load_env(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} configs: {df.status.value_counts().to_dict()}")

    fig_latency_vs_batch(df, env, args.out)
    fig_latency_vs_image_size(df, env, args.out)
    fig_latency_vs_pixels(df, env, args.out)
    fig_latency_spread(df, env, args.out)
    fig_throughput(df, env, args.out)
    fig_memory(df, env, args.out)
    fig_energy(df, env, args.out)
    fig_power(df, env, args.out)
    fig_heatmap(df, env, args.out, "latency_s", 1e3, "latency, мс", "Latency по сетке (S, B)",
                "grid_latency.png", lambda v: f"{v:.2f}" if v < 10 else f"{v:.0f}")
    fig_heatmap(df, env, args.out, "memory_bytes", 1 / 2**20, "peak memory, MiB",
                "Пиковая память по сетке (S, B)", "grid_memory.png",
                lambda v: f"{v:.0f}" if v < 1e4 else f"{v / 1024:.0f}G")
    fig_heatmap(df, env, args.out, "energy_j", 1.0, "энергия, Дж", "Энергия forward pass по сетке (S, B)",
                "grid_energy.png", lambda v: f"{v:.3f}" if v < 1 else f"{v:.1f}")


if __name__ == "__main__":
    main()
