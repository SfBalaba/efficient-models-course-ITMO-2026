"""Roofline-анализ: аппаратные константы GPU, ArI, T_compute / T_memory, энергия, пиковая память.

Сравнивает три набора коэффициентов:
  * идеальные  — из паспорта GPU (roofline: T = max(FLOPs / Peak, Bytes / BW), E = P_TDP · T);
  * подогнанные — results/theta.json (calibrate.py);
  * эффективные — посчитанные напрямую из замеров и трасс profile_model.py.

Формулы:
  T_compute = FLOPs / Peak FLOPS          T_memory = Bytes(I/O) / BW
  ArI       = FLOPs / Bytes(I/O)          Ridge    = Peak FLOPS / BW
  E         = P_TDP · T_real
Bytes(I/O) — минимальный трафик DRAM по слоям: чтение входа и весов + запись выхода
(ReLU in-place: чтение + запись, MaxPool: + запись int64-индексов).

Пиковая память — модель времени жизни тензоров: константа (веса + workspace cuBLAS)
+ входной тензор + максимум по слоям (вход слоя + выход слоя + индексы MaxPool).
Форма тензоров снимается forward-хуками на meta-устройстве (без GPU).

Выход:
  results/gpu_constants.json   идеальные коэффициенты и константы GPU
  results/coefficients.json    идеал / theta.json / эффективные + вердикт
  results/roofline.csv         метрики по каждой точке сетки
  results/roofline_layers.csv  per-layer roofline по трассам
  results/roofline_report.md   отчёт с вердиктом
  results/figures/roofline.png, results/figures/memory_model.png

Запуск:
    python roofline.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from equations import flops as flops_eq
from equations import memory as memory_eq
from models import Model
from plot_measurements import GRID, INK, INK_2, REFERENCE, SURFACE, save

HW_DIR = Path(__file__).resolve().parent
RESULTS = HW_DIR / "results"
FIGURES = RESULTS / "figures"
MiB = 2**20

# паспорт RTX 5070 Ti; источник — спецификация NVIDIA, в этом скрипте не проверяется
SPEC = {
    "gpu": "NVIDIA GeForce RTX 5070 Ti",
    "cuda_cores": 8960,
    "boost_clock_hz": 2.452e9,
    "mem_bandwidth_Bps": 896e9,
    "tdp_w": 300.0,
    "l2_cache_bytes": 48 * MiB,
    "source": "паспорт NVIDIA (не проверено по сети); L2 и TDP совпадают с torch/nvidia-smi на этой машине",
}
# workspace cuBLAS/cuBLASLt: первый nn.Linear на этой машине добавил 9,13 MiB к memory_allocated
CUBLAS_WORKSPACE_BYTES = int(9.125 * MiB)

LARGE = "S >= 368 & B >= 32"   # линейный участок: GPU насыщен, мощность упёрта в лимит
TRACE_CONFIGS = [(32, 1), (224, 32), (512, 64)]
CONFIG_STYLE = {(32, 1): ("#2a78d6", "o"), (224, 32): ("#eb6834", "s"), (512, 64): ("#1baf7a", "D")}


# --------------------------------------------------------------------------- per-layer shapes

_MODEL = None


def layer_table(S, B):
    """FLOPs, трафик и размеры тензоров каждого листового слоя (meta-устройство, без вычислений)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = Model().to("meta").eval()
    x = torch.empty(B, 3, S, S, device="meta")
    rows, handles = [], []

    def make_hook(name):
        def hook(mod, inp, out):
            t_in = inp[0]
            n_in, n_out = t_in.numel(), out.numel()
            w_bytes = 4 * sum(p.numel() for p in mod.parameters(recurse=False))
            inplace = out is t_in
            if isinstance(mod, torch.nn.Conv2d):
                k = mod.kernel_size[0] * mod.kernel_size[1]
                f = 2 * mod.in_channels // mod.groups * k * n_out
                io = 4 * (n_in + n_out) + w_bytes
                kind, idx = "conv", 0
            elif isinstance(mod, torch.nn.Linear):
                f = 2 * mod.in_features * n_out
                io = 4 * (n_in + n_out) + w_bytes
                kind, idx = "linear", 0
            elif isinstance(mod, torch.nn.MaxPool2d):
                f, idx = 0, 8 * n_out                      # int64-индексы (max_pool2d_with_indices на CUDA)
                io = 4 * (n_in + n_out) + idx
                kind = "maxpool"
            else:                                          # ReLU (in-place), AdaptiveAvgPool2d
                f = 0
                io = 4 * (n_in + n_out)
                kind, idx = "relu" if isinstance(mod, torch.nn.ReLU) else "avgpool", 0
            rows.append(dict(layer=name, kind=kind, flops=f, bytes_io=io,
                             in_bytes=4 * n_in, out_bytes=4 * n_out, idx_bytes=idx,
                             inplace=inplace, input_is_x=t_in is x))
        return hook

    for name, mod in _MODEL.named_modules():
        if not list(mod.children()):
            handles.append(mod.register_forward_hook(make_hook(name)))
    with torch.inference_mode():
        _MODEL(x)
    for h in handles:
        h.remove()
    return pd.DataFrame(rows)


def peak_memory_model(layers, S, B, const_bytes):
    """Константа + вход + max по слоям (вход слоя + новый выход + индексы MaxPool). Без workspace cuDNN."""
    stage = []
    for r in layers.itertuples():
        live_in = 0 if r.input_is_x else r.in_bytes
        live_out = 0 if r.inplace else r.out_bytes
        stage.append(live_in + live_out + r.idx_bytes)
    i = int(np.argmax(stage))
    return const_bytes + 4 * B * 3 * S * S + stage[i], layers.layer.iloc[i]


# --------------------------------------------------------------------------- constants

def gpu_constants(df, env):
    large = df.query(LARGE)
    clock = float(large.sm_clock_mhz.median()) * 1e6
    peak_spec = SPEC["cuda_cores"] * 2 * SPEC["boost_clock_hz"]
    peak = SPEC["cuda_cores"] * 2 * clock
    bw = SPEC["mem_bandwidth_Bps"]
    const = int((df.mem_before_bytes - 4 * df.B * 3 * df.S ** 2).min())
    return {
        "spec": SPEC,
        "measured": {
            "sm_clock_under_load_hz": clock,
            "power_limit_w": env.get("power_limit_w"),
            "idle_power_w": env.get("idle_power_w"),
            "params_bytes": env.get("param_bytes"),
            "cublas_workspace_bytes": CUBLAS_WORKSPACE_BYTES,
            "const_before_forward_bytes": const,
        },
        "derived": {
            "peak_fp32_flops_boost": peak_spec,
            "peak_fp32_flops_at_measured_clock": peak,
            "ridge_point_flop_per_byte": peak / bw,
        },
        "ideal_theta": {
            "theta_comp_s_per_flop": 1 / peak,
            "theta_mem_s_per_byte": 1 / bw,
            "theta_launch_s": 0.0,
            "theta_power_w_tdp": SPEC["tdp_w"],
            "theta_power_w_limit": env.get("power_limit_w"),
            "note": "roofline: T = max(FLOPs·theta_comp, Bytes·theta_mem), E = P·T; "
                    "theta_launch = 0 — в roofline нет накладных расходов",
        },
    }


# --------------------------------------------------------------------------- grid metrics

def grid_metrics(df, c):
    peak = c["derived"]["peak_fp32_flops_at_measured_clock"]
    bw = SPEC["mem_bandwidth_Bps"]
    tdp, p_lim = SPEC["tdp_w"], c["measured"]["power_limit_w"]
    const = c["measured"]["const_before_forward_bytes"] - 4 * 3 * 0  # константа уже без входа
    rows = []
    for r in df.itertuples():
        L = layer_table(r.S, r.B)
        F, IO = L.flops.sum(), L.bytes_io.sum()
        t_layers = np.maximum(L.flops / peak, L.bytes_io / bw).sum()
        mem_model, peak_layer = peak_memory_model(L, r.S, r.B, const)
        rows.append(dict(
            S=r.S, B=r.B, is_validation=r.is_validation,
            flops=F, flops_equations=flops_eq(r.S, r.B), bytes_io=IO, ArI=F / IO,
            t_compute_s=F / peak, t_memory_s=IO / bw,
            t_roofline_s=max(F / peak, IO / bw), t_roofline_layers_s=t_layers,
            latency_s=r.latency_s, roofline_efficiency=t_layers / r.latency_s,
            achieved_tflops=F / r.latency_s / 1e12, achieved_GBps=IO / r.latency_s / 1e9,
            energy_j=r.energy_j, energy_tdp_j=tdp * r.latency_s, energy_limit_j=p_lim * r.latency_s,
            avg_power_w=r.avg_power_w,
            memory_bytes=r.memory_bytes, memory_model_bytes=mem_model, memory_model_peak_layer=peak_layer,
            memory_equations_bytes=memory_eq(r.S, r.B),
            cudnn_workspace_bytes=r.memory_bytes - mem_model,
        ))
    return pd.DataFrame(rows)


def trace_layers(c):
    peak = c["derived"]["peak_fp32_flops_at_measured_clock"]
    bw = SPEC["mem_bandwidth_Bps"]
    out = []
    for S, B in TRACE_CONFIGS:
        path = RESULTS / "traces" / f"layers_S{S}_B{B}.csv"
        if not path.exists():
            continue
        t = pd.read_csv(path).set_index("layer")["device_time_us"]
        L = layer_table(S, B)
        L["S"], L["B"] = S, B
        L["t_measured_s"] = L.layer.map(t) * 1e-6
        L["ArI"] = L.flops / L.bytes_io
        L["t_compute_s"] = L.flops / peak
        L["t_memory_s"] = L.bytes_io / bw
        L["t_roofline_s"] = np.maximum(L.t_compute_s, L.t_memory_s)
        L["bound"] = np.where(L.ArI > c["derived"]["ridge_point_flop_per_byte"], "compute", "memory")
        L["efficiency"] = L.t_roofline_s / L.t_measured_s
        L["achieved_tflops"] = L.flops / L.t_measured_s / 1e12
        L["achieved_GBps"] = L.bytes_io / L.t_measured_s / 1e9
        L["time_share"] = L.t_measured_s / L.t_measured_s.sum()
        out.append(L)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def trace_memory_peaks():
    """Пик аллокаций за один шаг профайлера (над базой перед шагом) и слой, где он достигнут."""
    res = {}
    for S, B in TRACE_CONFIGS:
        path = RESULTS / "traces" / f"trace_S{S}_B{B}.json"
        if not path.exists():
            continue
        ev = json.loads(path.read_text())["traceEvents"]
        mem = sorted((e for e in ev if e.get("name") == "[memory]" and e["args"].get("Device Type") == 1),
                     key=lambda e: e["ts"])
        layers = [e for e in ev if e.get("ph") == "X" and e.get("name", "").startswith("LAYER/")]
        step = min((e for e in ev if e.get("ph") == "X" and e.get("name", "").startswith("ProfilerStep")),
                   key=lambda e: e["ts"])
        in_step = [e for e in mem if step["ts"] <= e["ts"] <= step["ts"] + step["dur"]]
        base = in_step[0]["args"]["Total Allocated"] - in_step[0]["args"]["Bytes"]
        top = max(in_step, key=lambda e: e["args"]["Total Allocated"])
        owner = [e["name"][6:] for e in layers if e["ts"] <= top["ts"] <= e["ts"] + e["dur"]]
        res[(S, B)] = dict(base=base, peak=top["args"]["Total Allocated"], layer=owner[0] if owner else "?")
    return res


# --------------------------------------------------------------------------- coefficients & verdict

def verdict(value, ideal, tol=1.5):
    if value is None or ideal is None:
        return "—"
    if ideal == 0:
        return "знак противоречит физике" if value < 0 else f"{value * 1e3:.3g} мс сверх roofline"
    if np.sign(value) != np.sign(ideal):
        return "знак противоречит физике"
    r = value / ideal
    if 1 / tol <= r <= tol:
        return f"согласуется (×{r:.2f})"
    return f"отклонение ×{r:.3g}"


def coefficients(df, grid, layers, c, theta):
    ideal = c["ideal_theta"]
    large = grid.query(LARGE)
    big_trace = layers[(layers.S == 512) & (layers.B == 64)] if len(layers) else pd.DataFrame()
    conv = big_trace[big_trace.kind == "conv"]
    ew = big_trace[big_trace.kind.isin(["relu", "maxpool", "avgpool"])]
    eff = {
        "theta_comp_s_per_flop": float(np.median(large.latency_s / large.flops)),
        "theta_comp_conv_only_s_per_flop": float(conv.t_measured_s.sum() / conv.flops.sum()) if len(conv) else None,
        "theta_mem_s_per_byte": float(ew.t_measured_s.sum() / ew.bytes_io.sum()) if len(ew) else None,
        "theta_launch_s": float(df.latency_s.min()),
        "theta_power_w": float(df.query(LARGE).avg_power_w.median()),
        "theta_power_w_small_inputs": float(df.query("S <= 64").avg_power_w.median()),
    }
    fit_lat = (theta or {}).get("latency", {})
    fit_en = (theta or {}).get("energy", {})
    table = [
        ("theta_comp, с/FLOP", ideal["theta_comp_s_per_flop"], fit_lat.get("theta_comp"), eff["theta_comp_s_per_flop"]),
        ("theta_comp (только свёртки), с/FLOP", ideal["theta_comp_s_per_flop"], None, eff["theta_comp_conv_only_s_per_flop"]),
        ("theta_mem, с/байт", ideal["theta_mem_s_per_byte"], fit_lat.get("theta_mem"), eff["theta_mem_s_per_byte"]),
        ("theta_launch, с", ideal["theta_launch_s"], fit_lat.get("theta_launch"), eff["theta_launch_s"]),
        ("theta_power, Вт (vs TDP)", ideal["theta_power_w_tdp"], fit_en.get("theta_power"), eff["theta_power_w"]),
        ("theta_power, Вт (vs лимит)", ideal["theta_power_w_limit"], fit_en.get("theta_power"), eff["theta_power_w"]),
    ]
    rows = [dict(coefficient=n, ideal=i, fitted_theta_json=f, effective_measured=e,
                 verdict_fitted=verdict(f, i), verdict_effective=verdict(e, i)) for n, i, f, e in table]
    return eff, rows


# --------------------------------------------------------------------------- figures

def fig_roofline(grid, layers, c, out):
    peak = c["derived"]["peak_fp32_flops_at_measured_clock"]
    peak_spec = c["derived"]["peak_fp32_flops_boost"]
    bw = SPEC["mem_bandwidth_Bps"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})

    ai = np.logspace(-1, 3.2, 200)
    ax.plot(ai, np.minimum(peak, ai * bw) / 1e12, color=INK, linewidth=1.6, zorder=1)
    ax.axhline(peak_spec / 1e12, color=REFERENCE, linestyle=":", linewidth=1.1)
    ax.axvline(c["derived"]["ridge_point_flop_per_byte"], color=REFERENCE, linestyle="--", linewidth=1.0)
    ax.annotate(f"пик {peak / 1e12:.1f} TFLOPS при {c['measured']['sm_clock_under_load_hz'] / 1e9:.2f} ГГц",
                (1500, peak / 1e12), xytext=(0, -13), textcoords="offset points", ha="right",
                color=INK_2, fontsize=8.5)
    ax.annotate(f"boost {SPEC['boost_clock_hz'] / 1e9:.2f} ГГц: {peak_spec / 1e12:.1f} TFLOPS",
                (1500, peak_spec / 1e12), xytext=(0, 5), textcoords="offset points", ha="right",
                color=INK_2, fontsize=8.5)
    ax.annotate(f"ridge {c['derived']['ridge_point_flop_per_byte']:.0f} FLOP/байт",
                (c["derived"]["ridge_point_flop_per_byte"], 2.0), xytext=(5, 0), textcoords="offset points",
                color=INK_2, fontsize=8.5)

    g = grid[grid.ArI > 0]
    ax.plot(g.ArI, g.achieved_tflops, "o", color=INK_2, markersize=3.5, alpha=0.6, zorder=2)
    handles = [Line2D([], [], color=INK, linewidth=1.6, label="roofline: min(Peak, ArI·BW)"),
               Line2D([], [], color=INK_2, marker="o", markersize=4, linestyle="none", alpha=0.6,
                      label="сеть целиком, 132 точки сетки")]
    for (S, B), (color, marker) in CONFIG_STYLE.items():
        sub = layers[(layers.S == S) & (layers.B == B) & (layers.flops > 0)]
        if sub.empty:
            continue
        ax.plot(sub.ArI, sub.achieved_tflops, marker, color=color, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
        handles.append(Line2D([], [], color=color, marker=marker, linestyle="none", markersize=7,
                              label=f"слои conv/fc, S={S}, B={B}"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.set_xlabel("арифметическая интенсивность ArI, FLOP/байт (log)")
    ax.set_ylabel("достигнутая производительность, TFLOPS (log)")
    ax.set_title("Roofline: свёртки и сеть целиком")
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)

    ew = layers[layers.kind.isin(["relu", "maxpool", "avgpool"])]
    names = list(dict.fromkeys(ew.layer))
    ypos = {n: i for i, n in enumerate(names)}
    for (S, B), (color, marker) in CONFIG_STYLE.items():
        sub = ew[(ew.S == S) & (ew.B == B)]
        ax2.plot(sub.achieved_GBps, [ypos[n] for n in sub.layer], marker, color=color, markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=0.8, linestyle="none", label=f"S={S}, B={B}")
    ax2.axvline(bw / 1e9, color=INK, linewidth=1.4, label=f"DRAM BW {bw / 1e9:.0f} GB/s")
    ax2.set_xscale("log")
    ax2.set_yticks(range(len(names)), names)
    ax2.invert_yaxis()
    ax2.set_xlabel("достигнутая пропускная способность, GB/s (log)")
    ax2.set_title("ReLU / MaxPool / GAP: 0 FLOP, memory-bound")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=8.5)
    save(fig, out, "roofline.png")


def fig_memory(grid, out):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    meas = grid.memory_bytes / MiB
    lo = min(meas.min(), grid.memory_equations_bytes.min() / MiB) / 1.5
    hi = max(meas.max(), grid.memory_equations_bytes.max() / MiB) * 1.5
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1.0)
    ax.plot(meas, grid.memory_equations_bytes / MiB, "o", color="#eb6834", markeredgecolor=SURFACE,
            markeredgewidth=0.8, label="equations.memory(): сумма всех активаций")
    ax.plot(meas, grid.memory_model_bytes / MiB, "o", color="#2a78d6", markeredgecolor=SURFACE,
            markeredgewidth=0.8, label="модель времени жизни (без workspace cuDNN)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("измерено max_memory_allocated, MiB")
    ax.set_ylabel("предсказано, MiB")
    ax.set_title("Пиковая память: две модели против замера")
    ax.legend(handles=[*ax.get_legend_handles_labels()[0],
                       Line2D([], [], color=INK_2, linewidth=1.0, label="предсказано = измерено")],
              loc="upper left", fontsize=8.5)
    save(fig, out, "memory_model.png")


# --------------------------------------------------------------------------- report

def fmt(v, unit=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.3g}{unit}"


def report(c, grid, layers, eff, coef_rows, trace_mem):
    peak = c["derived"]["peak_fp32_flops_at_measured_clock"]
    large = grid.query(LARGE)
    lines = ["# Roofline: константы, коэффициенты, вердикт", "",
             "Сгенерировано `python roofline.py`. Все числа — из `results/measurements.csv`, "
             "`results/traces/`, `results/theta.json`.", "",
             "## Константы GPU", "",
             "| | значение | источник |", "|---|---|---|",
             f"| Peak FP32 (boost {SPEC['boost_clock_hz'] / 1e9:.3f} ГГц) | {c['derived']['peak_fp32_flops_boost'] / 1e12:.1f} TFLOPS | паспорт |",
             f"| Peak FP32 при частоте под нагрузкой ({c['measured']['sm_clock_under_load_hz'] / 1e9:.3f} ГГц) | {peak / 1e12:.1f} TFLOPS | паспорт × замер частоты |",
             f"| BW DRAM | {SPEC['mem_bandwidth_Bps'] / 1e9:.0f} GB/s | паспорт |",
             f"| Ridge point | {c['derived']['ridge_point_flop_per_byte']:.1f} FLOP/байт | Peak / BW |",
             f"| TDP / лимит при замерах / idle | {SPEC['tdp_w']:.0f} / {c['measured']['power_limit_w']:.0f} / {c['measured']['idle_power_w']:.1f} W | nvidia-smi, env.json |",
             f"| Константа памяти до прохода | {c['measured']['const_before_forward_bytes'] / MiB:.2f} MiB | замер |",
             f"| — веса | {c['measured']['params_bytes'] / MiB:.2f} MiB | env.json |",
             f"| — workspace cuBLAS | {CUBLAS_WORKSPACE_BYTES / MiB:.2f} MiB | замер: первый nn.Linear |",
             "", "## Коэффициенты: идеал vs theta.json vs эффективные", "",
             "| коэффициент | идеал (GPU) | theta.json | эффективный (замер) | theta.json / идеал | эффективный / идеал |",
             "|---|---|---|---|---|---|"]
    for r in coef_rows:
        lines.append(f"| {r['coefficient']} | {fmt(r['ideal'])} | {fmt(r['fitted_theta_json'])} | "
                     f"{fmt(r['effective_measured'])} | {r['verdict_fitted']} | {r['verdict_effective']} |")

    lines += ["", "Как считались эффективные коэффициенты:",
              f"- theta_comp — медиана latency / FLOPs на линейном участке ({LARGE}); "
              f"1/theta = {1 / eff['theta_comp_s_per_flop'] / 1e12:.1f} TFLOPS;",
              f"- theta_comp (только свёртки) — Σt / ΣFLOPs свёрток по трассе S=512, B=64; "
              f"1/theta = {1 / eff['theta_comp_conv_only_s_per_flop'] / 1e12:.1f} TFLOPS;" if eff["theta_comp_conv_only_s_per_flop"] else "",
              f"- theta_mem — Σt / Σбайт ReLU/MaxPool/GAP по трассе S=512, B=64; "
              f"1/theta = {1 / eff['theta_mem_s_per_byte'] / 1e9:.0f} GB/s;" if eff["theta_mem_s_per_byte"] else "",
              f"- theta_launch — минимальная latency на сетке; в roofline накладных расходов нет, идеал = 0;",
              f"- theta_power — медиана средней мощности при {LARGE}; при S ≤ 64 — "
              f"{eff['theta_power_w_small_inputs']:.0f} W.", ""]

    # roofline по сети
    lines += ["## Roofline по сети целиком", "",
              f"- ArI сети: {grid.ArI.min():.1f}–{grid.ArI.max():.1f} FLOP/байт, "
              f"при {LARGE}: {large.ArI.min():.1f}–{large.ArI.max():.1f}; ridge = {c['derived']['ridge_point_flop_per_byte']:.1f}.",
              f"- T_measured / T_roofline (Σ max по слоям): {(1 / grid.roofline_efficiency).min():.1f}–"
              f"{(1 / grid.roofline_efficiency).max():.0f}×; при {LARGE}: "
              f"{(1 / large.roofline_efficiency).min():.2f}–{(1 / large.roofline_efficiency).max():.2f}×.",
              f"- Достигнуто: максимум {grid.achieved_tflops.max():.1f} TFLOPS = "
              f"{grid.achieved_tflops.max() * 1e12 / peak * 100:.0f} % пика.", ""]

    # per-layer
    if len(layers):
        lines += ["## Roofline по слоям (трассы)", "",
                  "Достигнуто и КПД — суммарно по группе: ΣFLOPs/Σt (или Σбайт/Σt) и Σt_roofline/Σt; "
                  "в скобках — диапазон по слоям с временем ≥ 5 мкс.", "",
                  "| конфиг | слои | доля времени | ArI, FLOP/байт | по roofline | достигнуто | КПД roofline |",
                  "|---|---|---|---|---|---|---|"]
        for (S, B) in TRACE_CONFIGS:
            sub = layers[(layers.S == S) & (layers.B == B)]
            if sub.empty:
                continue
            for kind, label in (("conv", "свёртки"), ("ew", "ReLU/MaxPool/GAP")):
                part = sub[sub.kind == "conv"] if kind == "conv" else sub[sub.kind.isin(["relu", "maxpool", "avgpool"])]
                t = part.t_measured_s.sum()
                big = part[part.t_measured_s >= 5e-6]
                eff_rng = f" ({big.efficiency.min():.2f}–{big.efficiency.max():.2f})" if len(big) else ""
                if kind == "conv":
                    rng = f" ({big.achieved_tflops.min():.1f}–{big.achieved_tflops.max():.1f})" if len(big) else ""
                    ach = f"{part.flops.sum() / t / 1e12:.2f} TFLOPS{rng}"
                    ari = f"{part.ArI.min():.0f}–{part.ArI.max():.0f}"
                    n_c = (part.bound == "compute").sum()
                    bound = f"compute: {n_c} из {len(part)}"
                else:
                    rng = f" ({big.achieved_GBps.min():.0f}–{big.achieved_GBps.max():.0f})" if len(big) else ""
                    ach = f"{part.bytes_io.sum() / t / 1e9:.0f} GB/s{rng}"
                    ari, bound = "0", "memory"
                lines.append(f"| S={S}, B={B} | {label} | {part.time_share.sum() * 100:.0f} % | {ari} | {bound} | "
                             f"{ach} | {part.t_roofline_s.sum() / t:.2f}{eff_rng} |")
        lines.append("")

    # память
    ok = (grid.cudnn_workspace_bytes.abs() / grid.memory_bytes < 1e-3).sum()
    over = (grid.cudnn_workspace_bytes < -1e-3 * grid.memory_bytes).sum()
    ratio_eq = grid.memory_equations_bytes / grid.memory_bytes
    lines += ["## Пиковая память", "",
              "Модель: константа (веса + workspace cuBLAS + прочее) + вход + max по слоям "
              "(вход слоя + выход слоя + int64-индексы MaxPool).", "",
              f"- совпадение с замером до 0,1 %: {ok} из {len(grid)} точек; модель выше замера: {over} точек;",
              f"- остаток (workspace cuDNN): до {grid.cudnn_workspace_bytes.max() / MiB:.0f} MiB "
              f"(S={int(grid.loc[grid.cudnn_workspace_bytes.idxmax(), 'S'])}, B={int(grid.loc[grid.cudnn_workspace_bytes.idxmax(), 'B'])});",
              f"- слой пика в модели: {', '.join(sorted(set(grid.memory_model_peak_layer)))};",
              f"- equations.memory() / замер: {ratio_eq.min():.2f}–{ratio_eq.max():.2f}.", ""]
    for (S, B), t in trace_mem.items():
        L = layer_table(S, B)
        p = L[L.kind == "maxpool"].iloc[0]
        c1 = L[L.layer == "conv1"].iloc[0]
        expect = c1.out_bytes + p.out_bytes + p.idx_bytes
        lines.append(f"- трасса S={S}, B={B}: пик над базой {(t['peak'] - t['base']) / MiB:.2f} MiB в слое {t['layer']}; "
                     f"conv1_out + pool_out + индексы = {expect / MiB:.2f} MiB")
    lines.append("")

    # энергия
    e_ratio = grid.energy_j / grid.energy_tdp_j
    e_ratio_lim = grid.energy_j / grid.energy_limit_j
    lines += ["## Энергия: E = P_TDP · T_real", "",
              f"- E_замер / (P_TDP · T): {e_ratio.min():.2f}–{e_ratio.max():.2f}, при {LARGE}: "
              f"{e_ratio[large.index].min():.2f}–{e_ratio[large.index].max():.2f};",
              f"- E_замер / (P_limit · T): при {LARGE}: {e_ratio_lim[large.index].min():.2f}–{e_ratio_lim[large.index].max():.2f};",
              f"- средняя мощность: {grid.avg_power_w.min():.0f}–{grid.avg_power_w.max():.0f} W.", ""]
    return "\n".join(line for line in lines if line is not None)


# --------------------------------------------------------------------------- main

def main():
    df = pd.read_csv(RESULTS / "measurements.csv")
    df = df[df.status == "OK"].reset_index(drop=True)
    env = json.loads((RESULTS / "env.json").read_text())
    theta_path = RESULTS / "theta.json"
    theta = json.loads(theta_path.read_text()) if theta_path.exists() else None

    c = gpu_constants(df, env)
    grid = grid_metrics(df, c)
    assert np.allclose(grid.flops, grid.flops_equations), "FLOPs по хукам не совпали с equations.flops"
    layers = trace_layers(c)
    trace_mem = trace_memory_peaks()
    eff, coef_rows = coefficients(df, grid, layers, c, theta)

    (RESULTS / "gpu_constants.json").write_text(json.dumps(c, indent=2, ensure_ascii=False))
    (RESULTS / "coefficients.json").write_text(json.dumps(
        {"ideal": c["ideal_theta"], "fitted_theta_json": theta, "effective_measured": eff, "table": coef_rows},
        indent=2, ensure_ascii=False))
    grid.to_csv(RESULTS / "roofline.csv", index=False)
    layers.to_csv(RESULTS / "roofline_layers.csv", index=False)
    text = report(c, grid, layers, eff, coef_rows, trace_mem)
    (RESULTS / "roofline_report.md").write_text(text)

    FIGURES.mkdir(parents=True, exist_ok=True)
    if len(layers):
        fig_roofline(grid, layers, c, FIGURES)
    fig_memory(grid, FIGURES)
    print(text)


if __name__ == "__main__":
    main()
