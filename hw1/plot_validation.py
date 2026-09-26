"""Валидация equations.py: предсказание и замер на одних графиках, по сетке (S, B).

Все файлы — в results/figures/validation/. Для каждой величины (FLOPs, latency, memory, energy):
  * <m>_vs_batch.png  — панель на каждый S: замеры (точки) и предсказание (кривая) vs B;
  * <m>_vs_size.png   — панель на каждый B: замеры и предсказание vs S;
  * <m>_surface.png   — поверхность предсказания над (S, B) и замеры точками;
  * <m>_error_grid.png — отношение предсказано / измерено в каждой клетке сетки.
Плюс parity.png — предсказано vs измерено для всех четырёх величин.

«Измеренные» FLOPs — счётчик torch.utils.flop_counter.FlopCounterMode (conv + addmm,
1 MAC = 2 FLOP), прогон на meta-устройстве без GPU. Latency / memory / energy — из
results/measurements.csv, theta — из results/theta.json (python calibrate.py).

Закрашенные маркеры — базовая сетка, полые — валидационные точки.

Запуск:
    python plot_validation.py
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

from equations import energy, flops, latency, memory
from plot_measurements import GRID, INK, INK_2, REFERENCE, SURFACE, load, load_env, log2_axis, save

HW_DIR = Path(__file__).resolve().parent

MEASURED = "#2a78d6"
PREDICTED = "#eb6834"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

B_TICKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
S_DENSE = np.arange(32, 513, 16)   # все допустимые S (кратные 16)
B_DENSE = np.arange(1, 257)

METRICS = {
    "flops": dict(col="flops_counted", scale=1e-9, unit="GFLOP", title="FLOPs",
                  pred=lambda S, B, th: flops(S, B)),
    "latency": dict(col="latency_s", scale=1e3, unit="latency, мс", title="Latency",
                    pred=lambda S, B, th: latency(S, B, th["latency"])),
    "memory": dict(col="memory_bytes", scale=1 / 2**20, unit="peak memory, MiB", title="Memory",
                   pred=lambda S, B, th: memory(S, B)),
    "energy": dict(col="energy_j", scale=1.0, unit="энергия, Дж", title="Energy",
                   pred=lambda S, B, th: energy(S, B, th["energy"])),
}


def predict(key, S, B, theta):
    cfg = METRICS[key]
    return np.asarray(cfg["pred"](S, B, theta), dtype=float) * cfg["scale"]


def positive(a):
    """Для лог-осей: неположительные предсказания (возможны при отрицательных theta) скрываем."""
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, a, np.nan)


def counted_flops(df):
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode

        from models import Model
    except ImportError as e:
        print(f"[warn] torch недоступен ({e}): графики FLOPs пропущены")
        return None
    model = Model().to("meta").eval()
    out = []
    for S, B in zip(df.S, df.B):
        x = torch.empty(int(B), 3, int(S), int(S), device="meta")
        with torch.inference_mode(), FlopCounterMode(display=False) as fc:
            model(x)
        out.append(fc.get_total_flops())
    return out


def mae(meas, pred):
    ok = np.isfinite(pred) & np.isfinite(meas)
    return float(np.mean(np.abs(pred[ok] - meas[ok])))


def point_legend(extra=()):
    return [
        Line2D([], [], color=MEASURED, marker="o", linestyle="none", markeredgecolor=SURFACE, label="замер, базовая сетка"),
        Line2D([], [], color=MEASURED, marker="o", markerfacecolor=SURFACE, linestyle="none",
               markeredgewidth=1.4, label="замер, валидация"),
        *extra,
    ]


def plot_points(ax, x, y, is_val):
    ax.plot(x[~is_val], y[~is_val], "o", color=MEASURED, markeredgecolor=SURFACE,
            markeredgewidth=0.8, zorder=3)
    ax.plot(x[is_val], y[is_val], "o", color=MEASURED, markerfacecolor=SURFACE,
            markeredgewidth=1.4, zorder=3)


def log_y(ax):
    """Лог-ось y с подписями 1-2-5: на панелях уже одной декады иначе остаётся один тик."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------------------- figures

def fig_small_multiples(df, env, theta, key, by, out):
    """Панель на каждое значение `by` (S или B); по оси x — другая переменная."""
    cfg = METRICS[key]
    x_col = "B" if by == "S" else "S"
    dense = B_DENSE if by == "S" else S_DENSE
    groups = sorted(df[by].unique())
    rand = set(env.get("random_S" if by == "S" else "random_B", ()))

    ncols = 4
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.7 * nrows), squeeze=False)
    for ax, g in zip(axes.flat, groups):
        sub = df[df[by] == g].sort_values(x_col)
        S_d, B_d = (g, dense) if by == "S" else (dense, g)
        ax.plot(dense, positive(predict(key, S_d, B_d, theta)), color=PREDICTED, linewidth=1.8, zorder=2)
        plot_points(ax, sub[x_col].to_numpy(), sub[cfg["col"]].to_numpy() * cfg["scale"],
                    sub.is_validation.to_numpy())
        log_y(ax)
        if x_col == "B":
            log2_axis(ax, [1, 4, 16, 64, 256])
        ax.set_title(f"{by} = {g}" + ("  (валидация)" if g in rand else ""), fontsize=10)

    handles = point_legend([Line2D([], [], color=PREDICTED, linewidth=1.8, label="предсказание")])
    spare = list(axes.flat)[len(groups):]
    for ax in spare:
        ax.axis("off")
    # легенда — в свободную ячейку сетки, а если её нет — под панелями
    if spare:
        spare[0].legend(handles=handles, loc="center", fontsize=9)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.supxlabel("batch size B" if x_col == "B" else "image size S, px", color=INK_2, fontsize=10,
                  y=0.035 if not spare else 0.0)
    fig.supylabel(f"{cfg['unit']} (log)", color=INK_2, fontsize=10)
    fig.suptitle(f"{cfg['title']}: замер и предсказание vs {x_col}, панель на каждый {by}",
                 color=INK, fontsize=12, weight="semibold")
    fig.tight_layout(rect=(0, 0.06 if not spare else 0, 1, 1))
    save(fig, out, f"{key}_vs_{'batch' if by == 'S' else 'size'}.png")


def fig_parity(df, theta, keys, out):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5))
    is_val = df.is_validation.to_numpy()
    for ax, key in zip(axes.flat, keys):
        cfg = METRICS[key]
        meas = df[cfg["col"]].to_numpy(dtype=float) * cfg["scale"]
        pred = positive(predict(key, df.S.to_numpy(), df.B.to_numpy(), theta))
        both = np.r_[meas, pred[np.isfinite(pred)]]
        lo, hi = both.min() / 1.5, both.max() * 1.5
        line = np.array([lo, hi])
        ax.plot(line, line, color=INK_2, linewidth=1.0, zorder=1)
        ax.plot(line, line * 1.25, ":", color=REFERENCE, linewidth=1.0, zorder=1)
        ax.plot(line, line / 1.25, ":", color=REFERENCE, linewidth=1.0, zorder=1)
        plot_points(ax, meas, pred, is_val)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"измерено, {cfg['unit']}")
        ax.set_ylabel(f"предсказано, {cfg['unit']}")
        unit = cfg["unit"].split(", ")[-1]
        ax.set_title(f"{cfg['title']}   MAE, {unit}: train {mae(meas[~is_val], pred[~is_val]):.3g}"
                     f" · val {mae(meas[is_val], pred[is_val]):.3g}", fontsize=10.5)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.legend(handles=point_legend([
        Line2D([], [], color=INK_2, linewidth=1.0, label="предсказано = измерено"),
        Line2D([], [], color=REFERENCE, linestyle=":", linewidth=1.0, label="±25 %"),
    ]), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Предсказание vs замер (parity)", color=INK, fontsize=12, weight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "parity.png")


def fig_error_grid(df, env, theta, key, out):
    """Клетка сетки = предсказано / измерено; цвет — log2 отношения (синий — занижение, красный — завышение)."""
    cfg = METRICS[key]
    s_vals, b_vals = sorted(df.S.unique()), sorted(df.B.unique())
    ratio = np.full((len(s_vals), len(b_vals)), np.nan)
    for _, r in df.iterrows():
        meas = r[cfg["col"]] * cfg["scale"]
        pred = predict(key, int(r.S), int(r.B), theta)
        ratio[s_vals.index(r.S), b_vals.index(r.B)] = pred / meas

    lim = 2.0  # ×4 в обе стороны
    color_val = np.clip(np.log2(np.where(ratio > 0, ratio, np.nan)), -lim, lim)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.grid(False)
    ax.set_facecolor(GRID)
    im = ax.imshow(color_val, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto", origin="lower")
    for i in range(len(s_vals)):
        for j in range(len(b_vals)):
            v = ratio[i, j]
            text = "≤0" if v <= 0 else f"{(v - 1) * 100:+.0f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=INK)
    rand_s, rand_b = set(env.get("random_S", ())), set(env.get("random_B", ()))
    ax.set_xticks(range(len(b_vals)), [f"{b}*" if b in rand_b else str(b) for b in b_vals])
    ax.set_yticks(range(len(s_vals)), [f"{s}*" if s in rand_s else str(s) for s in s_vals])
    ax.set_xlabel("batch size B   (* — валидационные значения)")
    ax.set_ylabel("image size S, px")
    ax.set_title(f"{cfg['title']}: ошибка предсказания, (предсказано − измерено) / измерено")
    cbar = fig.colorbar(im, ax=ax, pad=0.02, ticks=[-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels(["×1/4", "×1/2", "×1", "×2", "×4"])
    cbar.set_label("предсказано / измерено", color=INK_2)
    cbar.outline.set_visible(False)
    save(fig, out, f"{key}_error_grid.png")


def fig_surface(df, theta, key, out):
    cfg = METRICS[key]
    SS, BB = np.meshgrid(S_DENSE, np.unique(np.round(np.geomspace(1, 256, 33)).astype(int)), indexing="ij")
    Z = np.log10(positive(predict(key, SS, BB, theta)))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(np.log2(BB), SS, Z, color=PREDICTED, alpha=0.25, linewidth=0, antialiased=True)
    ax.plot_wireframe(np.log2(BB), SS, Z, rstride=4, cstride=4, color=PREDICTED, linewidth=0.6, alpha=0.8)

    meas = np.log10(df[cfg["col"]].to_numpy(dtype=float) * cfg["scale"])
    xb, ys, is_val = np.log2(df.B.to_numpy()), df.S.to_numpy(), df.is_validation.to_numpy()
    ax.scatter(xb[~is_val], ys[~is_val], meas[~is_val], color=MEASURED, s=18, depthshade=False)
    ax.scatter(xb[is_val], ys[is_val], meas[is_val], facecolors=SURFACE, edgecolors=MEASURED,
               linewidths=1.2, s=18, depthshade=False)

    ax.set_xticks(np.log2(B_TICKS), [str(b) for b in B_TICKS])
    z_all = np.r_[Z[np.isfinite(Z)], meas]
    ax.zaxis.set_major_locator(FixedLocator(np.arange(np.floor(z_all.min()), np.ceil(z_all.max()) + 1)))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{10 ** v:g}"))
    ax.set_xlabel("batch size B", color=INK_2)
    ax.set_ylabel("image size S, px", color=INK_2)
    ax.set_zlabel(f"{cfg['unit']} (log)", color=INK_2, labelpad=12)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(SURFACE)
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=22, azim=-128)
    ax.set_title(f"{cfg['title']}: поверхность предсказания и замеры", color=INK)
    ax.legend(handles=point_legend([Line2D([], [], color=PREDICTED, linewidth=1.8, label="предсказание")]),
              loc="upper left", fontsize=8.5)
    save(fig, out, f"{key}_surface.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=HW_DIR / "results" / "measurements.csv")
    p.add_argument("--theta", type=Path, default=HW_DIR / "results" / "theta.json")
    p.add_argument("--out", type=Path, default=HW_DIR / "results" / "figures" / "validation")
    args = p.parse_args()

    if not args.theta.exists():
        raise SystemExit(f"нет {args.theta} — сначала python calibrate.py")
    theta = json.loads(args.theta.read_text())
    env = load_env(args.csv)
    df = load(args.csv)
    df = df[df.status == "OK"].reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)

    keys = ["latency", "memory", "energy"]
    counted = counted_flops(df)
    if counted is not None:
        df["flops_counted"] = counted
        keys.insert(0, "flops")

    for key in keys:
        sub = df[df[METRICS[key]["col"]].notna()]
        fig_small_multiples(sub, env, theta, key, "S", args.out)
        fig_small_multiples(sub, env, theta, key, "B", args.out)
        fig_surface(sub, theta, key, args.out)
        fig_error_grid(sub, env, theta, key, args.out)
    fig_parity(df, theta, keys, args.out)


if __name__ == "__main__":
    main()
