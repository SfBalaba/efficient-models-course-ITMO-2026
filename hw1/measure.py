"""Замеры latency / peak memory / energy одного forward pass на сетке (S, B).

Протокол (home_work_one.md, разд. 4 и 7):
  * eval() + torch.inference_mode(), FP32, TF32 выключен, cudnn.benchmark = False;
  * вход — случайный тензор B x 3 x S x S;
  * latency — медиана wall-clock одного прохода (synchronize до и после),
    дополнительно пишется медиана GPU-времени по CUDA events;
  * memory — torch.cuda.max_memory_allocated() за один проход
    (включает веса модели и входной тензор);
  * energy — энергия всего GPU по счётчику NVML (мДж) на серии проходов,
    делённая на число проходов; если счётчик недоступен — интегрирование
    мощности, опрашиваемой в отдельном потоке;
  * OOM ловится и записывается как status = OOM.

Запуск:
    python measure.py                      # полная сетка -> results/measurements.csv
    python measure.py --resume             # досчитать недостающие точки
    python measure.py --quick --out /tmp/m.csv   # быстрая проверка
"""

import argparse
import csv
import gc
import json
import platform
import random
import statistics
import threading
import time
from pathlib import Path

import torch

from models import Model

HW_DIR = Path(__file__).resolve().parent
RESULTS_DIR = HW_DIR / "results"

BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
N_RANDOM_S = 4
N_RANDOM_B = 3

CSV_FIELDS = [
    "S", "B", "is_validation", "status",
    "latency_s",            # медиана wall-clock одного прохода
    "memory_bytes",         # max_memory_allocated за проход
    "energy_j",             # энергия GPU на один проход
    "latency_p10_s", "latency_p90_s", "latency_mean_s", "latency_std_s",
    "latency_gpu_s",        # медиана по CUDA events (без учёта простоя CPU до старта)
    "n_reps",
    "mem_before_bytes",     # веса + вход до прохода
    "mem_activation_bytes", # memory_bytes - mem_before_bytes
    "energy_iters", "energy_window_s", "avg_power_w", "energy_method",
    "sm_clock_mhz", "temperature_c",
    "error",
]


# --------------------------------------------------------------------------- grid

def sample_grid(seed):
    """4 случайных S (кратных 16, вне базовой сетки) и 3 случайных B (не степени двойки)."""
    rng = random.Random(seed)
    s_pool = [s for s in range(32, 513, 16) if s not in BASE_S]
    b_pool = [b for b in range(1, 257) if b & (b - 1) != 0]
    return sorted(rng.sample(s_pool, N_RANDOM_S)), sorted(rng.sample(b_pool, N_RANDOM_B))


def build_grid(seed, quick=False):
    if quick:
        return [(s, b, False) for s in (32, 224) for b in (1, 8)], [], []
    rand_s, rand_b = sample_grid(seed)
    all_s = sorted(BASE_S + rand_s)
    all_b = sorted(BASE_B + rand_b)
    grid = [(s, b, s in rand_s or b in rand_b) for s in all_s for b in all_b]
    return grid, rand_s, rand_b


# --------------------------------------------------------------------------- NVML

class GpuMonitor:
    """Энергия / мощность / частоты через NVML (пакет nvidia-ml-py)."""

    def __init__(self, device_index):
        self.nvml = None
        self.handle = None
        self.has_counter = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:  
            print(f"[warn] NVML недоступен ({e}); энергия не будет измерена")
            return
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.has_counter = True
        except Exception:  
            print("[warn] счётчик энергии NVML не поддерживается, будет интегрироваться мощность")

    @property
    def available(self):
        return self.handle is not None

    def energy_mj(self):
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)

    def power_w(self):
        return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def sm_clock_mhz(self):
        return self.nvml.nvmlDeviceGetClockInfo(self.handle, self.nvml.NVML_CLOCK_SM)

    def temperature_c(self):
        return self.nvml.nvmlDeviceGetTemperature(self.handle, self.nvml.NVML_TEMPERATURE_GPU)

    def info(self):
        if not self.available:
            return {}
        out = {}
        for key, fn in {
            "driver_version": lambda: self.nvml.nvmlSystemGetDriverVersion(),
            "power_limit_w": lambda: self.nvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0,
            "max_sm_clock_mhz": lambda: self.nvml.nvmlDeviceGetMaxClockInfo(self.handle, self.nvml.NVML_CLOCK_SM),
            "max_mem_clock_mhz": lambda: self.nvml.nvmlDeviceGetMaxClockInfo(self.handle, self.nvml.NVML_CLOCK_MEM),
        }.items():
            try:
                val = fn()
                out[key] = val.decode() if isinstance(val, bytes) else val
            except Exception:  # noqa: BLE001
                out[key] = None
        out["energy_counter"] = self.has_counter
        return out

    def idle_power_w(self, seconds=2.0, interval=0.02):
        torch.cuda.synchronize()
        time.sleep(0.5)
        samples = []
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            samples.append(self.power_w())
            time.sleep(interval)
        return statistics.median(samples)


class PowerSampler(threading.Thread):
    """Запасной вариант: интегрирует мощность методом трапеций."""

    def __init__(self, monitor, interval=0.005):
        super().__init__(daemon=True)
        self.monitor = monitor
        self.interval = interval
        self.samples = []
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            self.samples.append((time.perf_counter(), self.monitor.power_w()))
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join()
        s = self.samples
        return sum((t1 - t0) * (p0 + p1) / 2 for (t0, p0), (t1, p1) in zip(s, s[1:]))


# --------------------------------------------------------------------------- measurements

def measure_memory(model, x):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.inference_mode():
        y = model(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del y
    return peak, before


def measure_latency(model, x, warmup, reps, max_seconds):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()

        wall, gpu = [], []
        t_budget = time.perf_counter() + max_seconds
        for i in range(reps):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            wall.append(time.perf_counter() - t0)
            gpu.append(start.elapsed_time(end) / 1e3)
            if i >= 9 and time.perf_counter() > t_budget:
                break

    q = statistics.quantiles(wall, n=10) if len(wall) >= 2 else [wall[0]] * 9
    return {
        "latency_s": statistics.median(wall),
        "latency_p10_s": q[0],
        "latency_p90_s": q[-1],
        "latency_mean_s": statistics.fmean(wall),
        "latency_std_s": statistics.stdev(wall) if len(wall) > 1 else 0.0,
        "latency_gpu_s": statistics.median(gpu),
        "n_reps": len(wall),
    }


def measure_energy(model, x, monitor, latency_s, min_seconds, min_iters):
    """Гоняем проходы подряд не меньше min_seconds и считаем энергию на один проход."""
    if not monitor.available:
        return {"energy_method": "none"}

    chunk = max(1, int(0.05 / max(latency_s, 1e-6)))  # синхронизация примерно раз в 50 мс
    n = 0
    with torch.inference_mode():
        torch.cuda.synchronize()
        if monitor.has_counter:
            e0 = monitor.energy_mj()
        else:
            sampler = PowerSampler(monitor)
            sampler.start()
        t0 = time.perf_counter()
        while True:
            for _ in range(chunk):
                model(x)
            n += chunk
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if elapsed >= min_seconds and n >= min_iters:
                break
        clock = monitor.sm_clock_mhz()
        temp = monitor.temperature_c()
        if monitor.has_counter:
            joules = (monitor.energy_mj() - e0) / 1e3
            method = "nvml_counter"
        else:
            joules = sampler.stop()
            method = "power_sampling"

    return {
        "energy_j": joules / n,
        "energy_iters": n,
        "energy_window_s": elapsed,
        "avg_power_w": joules / elapsed,
        "energy_method": method,
        "sm_clock_mhz": clock,
        "temperature_c": temp,
    }


def is_oom(err):
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    msg = str(err).lower()
    return "out of memory" in msg or "cudnn_status_alloc_failed" in msg


def cleanup():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def run_config(model, S, B, device, monitor, args):
    row = {"S": S, "B": B}
    x = None
    try:
        x = torch.randn(B, 3, S, S, device=device)
        peak, before = measure_memory(model, x)
        row.update(memory_bytes=peak, mem_before_bytes=before, mem_activation_bytes=peak - before)
        row.update(measure_latency(model, x, args.warmup, args.reps, args.max_latency_seconds))
        row.update(measure_energy(model, x, monitor, row["latency_s"],
                                  args.energy_seconds, args.energy_min_iters))
        row["status"] = "OK"
    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
        row["status"] = "OOM" if is_oom(err) else "ERROR"
        row["error"] = str(err).splitlines()[0][:200]
    finally:
        del x
        cleanup()
    return row


# --------------------------------------------------------------------------- main

def environment_info(model, device, monitor, idle_power, args, rand_s, rand_b):
    props = torch.cuda.get_device_properties(device)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "gpu": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        **monitor.info(),
        "idle_power_w": idle_power,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "n_params": n_params,
        "param_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "grid_seed": args.seed,
        "random_S": rand_s,
        "random_B": rand_b,
        "protocol": {
            "warmup": args.warmup,
            "reps": args.reps,
            "max_latency_seconds": args.max_latency_seconds,
            "energy_seconds": args.energy_seconds,
            "energy_min_iters": args.energy_min_iters,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "dtype": "float32",
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "measurements.csv")
    p.add_argument("--env-out", type=Path, default=None, help="по умолчанию env.json рядом с --out")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026, help="seed для случайных S и B")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--max-latency-seconds", type=float, default=5.0,
                   help="лимит времени на замер latency одной точки (минимум 10 повторов)")
    p.add_argument("--energy-seconds", type=float, default=2.0,
                   help="минимальная длительность окна замера энергии")
    p.add_argument("--energy-min-iters", type=int, default=20)
    p.add_argument("--resume", action="store_true", help="пропустить точки, уже записанные в --out")
    p.add_argument("--quick", action="store_true", help="4 точки для проверки работоспособности")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA недоступна — нужен PyTorch со сборкой под CUDA")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = Model().to(device).eval()

    monitor = GpuMonitor(device.index or 0)
    idle_power = monitor.idle_power_w() if monitor.available else None

    grid, rand_s, rand_b = build_grid(args.seed, args.quick)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    env_out = args.env_out or args.out.with_name("env.json")
    env = environment_info(model, device, monitor, idle_power, args, rand_s, rand_b)
    env_out.write_text(json.dumps(env, indent=2, ensure_ascii=False))
    print(f"{env['gpu']} | torch {env['torch']} | CUDA {env['cuda']} | idle {idle_power} W")
    print(f"random S = {rand_s}, random B = {rand_b}, {len(grid)} configs")

    # прогрев GPU (частоты, cuDNN-хэндлы), чтобы первая точка сетки не была «холодной»
    with torch.inference_mode():
        x = torch.randn(16, 3, 224, 224, device=device)
        t_end = time.perf_counter() + 3.0
        while time.perf_counter() < t_end:
            model(x)
            torch.cuda.synchronize()
        del x
    cleanup()

    done = set()
    if args.resume and args.out.exists():
        with args.out.open() as f:
            done = {(int(r["S"]), int(r["B"])) for r in csv.DictReader(f)}
    mode = "a" if args.resume and args.out.exists() else "w"

    with args.out.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
        for i, (S, B, is_val) in enumerate(grid, 1):
            if (S, B) in done:
                continue
            row = run_config(model, S, B, device, monitor, args)
            row["is_validation"] = int(is_val)
            writer.writerow(row)
            f.flush()

            if row["status"] == "OK":
                msg = (f"lat {row['latency_s'] * 1e3:9.3f} ms | mem {row['memory_bytes'] / 2**20:9.1f} MiB"
                       + (f" | E {row['energy_j']:.4f} J @ {row['avg_power_w']:.0f} W" if "energy_j" in row else ""))
            else:
                msg = f"{row['status']}: {row.get('error', '')[:80]}"
            print(f"[{i:3d}/{len(grid)}] S={S:3d} B={B:3d}{' (val)' if is_val else '      '} {msg}", flush=True)

    print(f"saved -> {args.out}\nenv   -> {env_out}")


if __name__ == "__main__":
    main()
