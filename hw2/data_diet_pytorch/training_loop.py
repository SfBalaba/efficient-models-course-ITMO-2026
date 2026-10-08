"""Fixed-step SGD training exactly as in the Data Diet paper, with exhaustive metric collection.

Key property of the paper: the NUMBER OF STEPS does not depend on the training-set size, so a model trained on a
pruned subset costs the same as one trained on all data. Examples are drawn without replacement, the incomplete
tail of a pass is dropped and the set is reshuffled (same as `train_batches` in the original code).

Extension points (so that score collection and pruning never require editing this file):
    step_observers : objects with observe_step(batch_indices, logits, labels) and summarize_epoch() -> dict
    score_callback : function(epoch, model, recorder) -> dict of extra metrics, called at `score_epochs`
    train_indices  : subset of the training set to train on (None = all examples)
"""
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from cifar10_data import CLASS_NAMES, NUM_CLASSES, Cifar10Data, augment_like_official
from resnet18_lowres import ResNet18LowRes, count_parameters
from run_logging import RunRecorder

STEPS_PER_EPOCH = 390  # 50000 // 128, the "epoch" length used for logging in the original code
MODEL_SEED_BASE, TRAIN_SEED_BASE, SEED_STRIDE = 42, 4242, 424242  # as in scripts/run_full_data.py
MIN_FREE_DISK_GB = 3.0


@dataclass
class TrainingConfig:
    run_number: int
    total_epochs: int = 200
    stop_after_epochs: Optional[int] = None  # stop early but keep the schedule of `total_epochs` (smoke tests)
    batch_size: int = 128
    precision: str = "bf16"  # "bf16", "tf32" or "fp32"
    learning_rate: float = 0.1
    momentum: float = 0.9
    use_nesterov: bool = True
    weight_decay: float = 5e-4
    lr_decay_factor: float = 0.2
    lr_decay_at_fraction_of_training: Sequence[float] = (0.3, 0.6, 0.8)  # epochs 60, 120, 160 of 200
    checkpoint_epochs: Sequence[int] = (0, 4, 8, 10, 12, 16, 20, 50, 100, 200)
    score_epochs: Sequence[int] = ()
    extra_parameters: dict = field(default_factory=dict)

    @property
    def model_seed(self) -> int:
        return MODEL_SEED_BASE + self.run_number * SEED_STRIDE

    @property
    def train_seed(self) -> int:
        return TRAIN_SEED_BASE + self.run_number * SEED_STRIDE

    @property
    def total_steps(self) -> int:
        return self.total_epochs * STEPS_PER_EPOCH

    @property
    def last_epoch(self) -> int:
        return min(self.total_epochs, self.stop_after_epochs or self.total_epochs)

    def learning_rate_at(self, step: int) -> float:
        decays = sum(step >= fraction * self.total_steps for fraction in self.lr_decay_at_fraction_of_training)
        return self.learning_rate * self.lr_decay_factor ** decays


@dataclass
class TrainingResult:
    model: torch.nn.Module
    epoch_rows: list
    train_seconds: float
    run_directory: Path
    manifest: dict


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, batch_size: int = 1000) -> dict:
    """Loss, accuracy, per-class accuracy and raw logits on a labelled set (eval-mode BatchNorm, no augmentation)."""
    model.eval()
    logits = torch.cat([model(images[start:start + batch_size].contiguous(memory_format=torch.channels_last)).float()
                        for start in range(0, len(images), batch_size)])
    model.train()
    correct = logits.argmax(1) == labels
    per_class_total = torch.bincount(labels, minlength=NUM_CLASSES).float()
    per_class_correct = torch.bincount(labels[correct], minlength=NUM_CLASSES).float()
    return {
        "loss": F.cross_entropy(logits, labels).item(),
        "accuracy": correct.float().mean().item(),
        "per_class_accuracy": (per_class_correct / per_class_total).cpu().numpy(),
        "logits": logits.cpu().numpy(),
    }


def _global_norm(tensors: list) -> torch.Tensor:
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(tensors)))


def _check_free_disk(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 2**30
    if free_gb < MIN_FREE_DISK_GB:
        raise RuntimeError(f"only {free_gb:.1f} GB free on the disk of {path}, need at least {MIN_FREE_DISK_GB} GB")


def train_with_fixed_steps(data: Cifar10Data, config: TrainingConfig, run_directory: Path,
                           train_indices: Optional[torch.Tensor] = None, step_observers: Sequence = (),
                           score_callback: Optional[Callable] = None, mlflow_experiment: Optional[str] = None,
                           mlflow_run_name: Optional[str] = None, mlflow_tags: Optional[dict] = None) -> TrainingResult:
    device = "cuda"
    run_directory = Path(run_directory)
    _check_free_disk(run_directory)
    num_all = data.num_train_examples
    train_indices = torch.arange(num_all, device=device) if train_indices is None else train_indices.to(device)
    num_training_examples = len(train_indices)
    keep_fraction = num_training_examples / num_all

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = config.precision != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = config.precision != "fp32"
    torch.manual_seed(config.model_seed)
    model = ResNet18LowRes(NUM_CLASSES).to(device).to(memory_format=torch.channels_last)
    parameters = list(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=config.learning_rate, momentum=config.momentum,
                                nesterov=config.use_nesterov, weight_decay=config.weight_decay)
    shuffle_generator = torch.Generator(device=device).manual_seed(config.train_seed)
    crop_offset_rng = np.random.RandomState(config.train_seed)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.precision == "bf16")
    checkpoint_epochs = set(config.checkpoint_epochs) | {config.last_epoch}
    score_epochs = set(config.score_epochs)

    run_parameters = {
        **{key: value for key, value in asdict(config).items() if key != "extra_parameters"},
        "model_seed": config.model_seed, "train_seed": config.train_seed, "total_steps": config.total_steps,
        "steps_per_epoch": STEPS_PER_EPOCH, "num_model_parameters": count_parameters(model),
        "lr_decay_steps": [int(f * config.total_steps) for f in config.lr_decay_at_fraction_of_training],
        "training_examples": num_training_examples, "keep_fraction": round(keep_fraction, 6),
        "prune_fraction": round(1 - keep_fraction, 6), "cifar10_train_data_sha256": data.train_data_sha256,
        "augmentation": "reflect-pad4, one crop offset per batch, per-image flip", "cudnn_benchmark": True,
        **config.extra_parameters,
    }
    recorder = RunRecorder(run_directory, run_parameters, mlflow_experiment, mlflow_run_name, mlflow_tags)
    try:
        result = _run_training(data, config, model, optimizer, parameters, train_indices, step_observers,
                               score_callback, recorder, shuffle_generator, crop_offset_rng, autocast,
                               checkpoint_epochs, score_epochs)
    except BaseException:
        recorder.finish(status="failed")
        raise
    manifest = recorder.finish("complete", {"final_test_accuracy": result[0][-1]["test_accuracy"],
                                            "total_train_seconds": result[1]})
    return TrainingResult(model, result[0], result[1], run_directory, manifest)


def _save_checkpoint(model, recorder, epoch, step, run_number, test_logits) -> None:
    checkpoint_path = f"checkpoints/model_epoch_{epoch:03d}.pt"
    logits_path = f"test_logits/test_logits_epoch_{epoch:03d}.npy"
    (recorder.run_directory / "checkpoints").mkdir(exist_ok=True)
    (recorder.run_directory / "test_logits").mkdir(exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "step": step, "run_number": run_number},
               recorder.run_directory / checkpoint_path)
    np.save(recorder.run_directory / logits_path, test_logits.astype(np.float16))
    recorder.expect_file(checkpoint_path)
    recorder.expect_file(logits_path)


def _run_training(data, config, model, optimizer, parameters, train_indices, step_observers, score_callback,
                  recorder, shuffle_generator, crop_offset_rng, autocast, checkpoint_epochs, score_epochs):
    device = "cuda"
    num_training_examples = len(train_indices)

    def record_epoch(epoch: int, step: int, learning_rate: float, train_stats: dict, train_seconds: float) -> dict:
        hook_start = time.perf_counter()
        test = evaluate_model(model, data.test_images, data.test_labels)
        row = {"epoch": epoch, "step": step, "learning_rate": learning_rate, **train_stats,
               "test_loss": test["loss"], "test_accuracy": test["accuracy"],
               **{f"test_accuracy_{name}": float(acc) for name, acc in zip(CLASS_NAMES, test["per_class_accuracy"])},
               "weight_norm": _global_norm([p.detach() for p in parameters]).item()}
        for observer in step_observers:
            row.update(observer.summarize_epoch())
        if epoch in checkpoint_epochs:
            _save_checkpoint(model, recorder, epoch, step, config.run_number, test["logits"])
        if epoch in score_epochs and score_callback is not None:
            row.update(score_callback(epoch, model, recorder) or {})
            model.train()
        torch.cuda.synchronize()
        row.update(epoch_train_seconds=train_seconds, epoch_eval_and_hooks_seconds=time.perf_counter() - hook_start,
                   peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / 2**30,
                   elapsed_minutes=(time.perf_counter() - run_start) / 60)
        recorder.log_epoch(epoch, row)
        return row

    model.train()
    rows = []
    torch.cuda.synchronize()
    run_start = time.perf_counter()
    rows.append(record_epoch(0, 0, config.learning_rate_at(1), {}, 0.0))
    loss_sum = torch.zeros((), device=device)
    accuracy_sum = torch.zeros((), device=device)
    gradient_norm_sum = torch.zeros((), device=device)
    steps_in_epoch, step, position, order = 0, 0, 0, None
    torch.cuda.reset_peak_memory_stats()
    epoch_start = time.perf_counter()
    while step < config.last_epoch * STEPS_PER_EPOCH:
        if order is None or position + config.batch_size > num_training_examples:
            order = train_indices[torch.randperm(num_training_examples, device=device, generator=shuffle_generator)]
            position = 0
        batch_indices = order[position:position + config.batch_size]
        position += config.batch_size
        step += 1
        images = augment_like_official(data.train_images[batch_indices], crop_offset_rng, shuffle_generator)
        labels = data.train_labels[batch_indices]
        learning_rate = config.learning_rate_at(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with autocast:
            logits = model(images.contiguous(memory_format=torch.channels_last))
        loss = F.cross_entropy(logits.float(), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            gradient_norm_sum += _global_norm([p.grad for p in parameters])
            loss_sum += loss.detach()
            accuracy_sum += (logits.argmax(1) == labels).float().mean()
            for observer in step_observers:
                observer.observe_step(batch_indices, logits.detach(), labels)
        optimizer.step()
        steps_in_epoch += 1
        if step % STEPS_PER_EPOCH == 0:
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - epoch_start
            train_stats = {"train_loss": (loss_sum / steps_in_epoch).item(),
                           "train_accuracy": (accuracy_sum / steps_in_epoch).item(),
                           "gradient_norm_mean": (gradient_norm_sum / steps_in_epoch).item()}
            if not np.isfinite(train_stats["train_loss"]):
                raise RuntimeError(f"non-finite training loss at step {step}")
            row = record_epoch(step // STEPS_PER_EPOCH, step, learning_rate, train_stats, train_seconds)
            rows.append(row)
            print(f"epoch {row['epoch']:3d}/{config.last_epoch} | {row['elapsed_minutes']:5.1f} min | lr {learning_rate:.4f}"
                  f" | train loss {row['train_loss']:.3f} acc {row['train_accuracy']:.3f}"
                  f" | test acc {row['test_accuracy']:.4f}", flush=True)
            for tensor in (loss_sum, accuracy_sum, gradient_norm_sum):
                tensor.zero_()
            steps_in_epoch = 0
            torch.cuda.reset_peak_memory_stats()
            epoch_start = time.perf_counter()
    return rows, time.perf_counter() - run_start
