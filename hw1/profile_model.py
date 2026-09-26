"""Профилирование одного (S, B) через torch.profiler — как в week_one/seminar.

Каждый слой модели оборачивается в record_function("LAYER/<name>"), поэтому
в Perfetto (https://ui.perfetto.dev) и в таблице видно время по слоям.

Запуск:
    python profile_model.py --S 224 --B 32
    python profile_model.py --S 32 --B 1        # launch-bound
    python profile_model.py --S 512 --B 64      # compute/memory-bound

Результаты: results/traces/trace_S{S}_B{B}.json, ops_S{S}_B{B}.txt, layers_S{S}_B{B}.csv
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from models import Model

TRACE_DIR = Path(__file__).resolve().parent / "results" / "traces"


def add_layer_ranges(model):
    """Навешивает record_function на каждый листовой модуль через forward hooks."""
    handles = []
    for name, module in model.named_modules():
        if list(module.children()):
            continue

        def pre_hook(m, inp, name=name):
            m._prof_range = record_function(f"LAYER/{name}")
            m._prof_range.__enter__()

        def post_hook(m, inp, out):
            m._prof_range.__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    return handles


def device_time_us(evt):
    # в новых версиях torch cuda_time_* переименованы в device_time_*
    return getattr(evt, "device_time_total", None) or getattr(evt, "cuda_time_total", 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--S", type=int, default=224)
    p.add_argument("--B", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", type=Path, default=TRACE_DIR)
    args = p.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    model = Model().to(args.device).eval()
    add_layer_ranges(model)
    x = torch.randn(args.B, 3, args.S, args.S, device=args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"S{args.S}_B{args.B}"
    trace_path = args.out_dir / f"trace_{tag}.json"

    # skip_first + repeat * [wait + warmup + active]
    sched = schedule(skip_first=1, wait=1, warmup=1, active=3, repeat=1)
    with torch.inference_mode(), profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace_path)),
    ) as prof:
        for _ in range(7):
            model(x)
            torch.cuda.synchronize()
            prof.step()

    events = prof.key_averages()
    table = events.table(sort_by="cuda_time_total", row_limit=40)
    (args.out_dir / f"ops_{tag}.txt").write_text(table)
    print(table)

    # время по слоям (среднее на один активный шаг)
    active_steps = 3
    # каждая аннотация встречается дважды: CPU-диапазон и его GPU-копия (gpu_user_annotation)
    layers = {}
    for e in events:
        if e.key.startswith("LAYER/"):
            cpu, dev = layers.get(e.key, (0.0, 0.0))
            layers[e.key] = (max(cpu, e.cpu_time_total), max(dev, device_time_us(e)))
    with (args.out_dir / f"layers_{tag}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "cpu_time_us", "device_time_us"])
        for key, (cpu, dev) in sorted(layers.items(), key=lambda kv: -kv[1][1]):
            w.writerow([key.removeprefix("LAYER/"), round(cpu / active_steps, 3), round(dev / active_steps, 3)])

    print(f"\ntrace  -> {trace_path}  (открыть в https://ui.perfetto.dev)")
    print(f"layers -> {args.out_dir / f'layers_{tag}.csv'}")


if __name__ == "__main__":
    main()
