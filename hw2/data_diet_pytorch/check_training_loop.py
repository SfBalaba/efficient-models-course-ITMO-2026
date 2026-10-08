"""Stage-1 check of the training loop: python check_training_loop.py

Runs 2 epochs of the real 200-epoch schedule on all data and 1 epoch on a 20% subset, then verifies:
output files and manifest, hooks (observer, score callback), mlflow parameters (keep/prune fraction),
and agreement with the earlier PyTorch run in hw2/dd_torch (same seeds => identical initial model)."""
import json
import shutil
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch

from cifar10_data import CLASS_NAMES, load_cifar10
from run_logging import MLFLOW_TRACKING_URI, verify_run_directory
from training_loop import STEPS_PER_EPOCH, TrainingConfig, train_with_fixed_steps

CHECK_DIRECTORY = Path(__file__).resolve().parent / "exps" / "stage1_checks"
PREVIOUS_RUN_HISTORY = Path(__file__).resolve().parents[1] / "dd_torch/exps/cifar10_resnet18/run_1/history.json"
failures = []


def check(description: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {description} {detail}")
    if not condition:
        failures.append(description)


class CountingObserver:
    def __init__(self):
        self.calls = 0
        self.seen_indices = set()

    def observe_step(self, batch_indices, logits, labels):
        self.calls += 1
        self.seen_indices.update(batch_indices.tolist()[:4])

    def summarize_epoch(self) -> dict:
        return {"observer_calls_total": float(self.calls)}


def score_callback(epoch, model, recorder):
    return {"callback_epoch_seen": float(epoch)}


data = load_cifar10()
shutil.rmtree(CHECK_DIRECTORY, ignore_errors=True)

# 1. all data, first 2 epochs of the real 200-epoch schedule
observer = CountingObserver()
config = TrainingConfig(run_number=1, stop_after_epochs=2, checkpoint_epochs=(0, 1, 2), score_epochs=(0, 2))
result = train_with_fixed_steps(data, config, CHECK_DIRECTORY / "all_data", step_observers=[observer],
                                score_callback=score_callback, mlflow_experiment="stage1-checks",
                                mlflow_run_name="check/all_data")
rows = result.epoch_rows
check("3 epoch rows (epochs 0, 1, 2)", [row["epoch"] for row in rows] == [0, 1, 2])
check("observer called once per step", observer.calls == 2 * STEPS_PER_EPOCH, f"({observer.calls})")
check("score callback only at score epochs", "callback_epoch_seen" in rows[0] and "callback_epoch_seen" not in rows[1]
      and rows[2]["callback_epoch_seen"] == 2.0)
check("per-class test accuracy for 10 classes", all(f"test_accuracy_{name}" in rows[1] for name in CLASS_NAMES))
check("extended metrics present", all(key in rows[1] for key in
      ("gradient_norm_mean", "weight_norm", "peak_gpu_memory_gb", "epoch_train_seconds", "learning_rate")))
check("manifest complete", result.manifest["status"] == "complete", str(result.manifest["problems"]))
check("verify_run_directory (with hashes) finds no problems",
      verify_run_directory(CHECK_DIRECTORY / "all_data", check_hashes=True) == [])
check("checkpoints and test logits saved for epochs 0, 1, 2", all(
    (CHECK_DIRECTORY / "all_data" / sub / name).is_file() for sub, name in
    [("checkpoints", f"model_epoch_{e:03d}.pt") for e in (0, 1, 2)] +
    [("test_logits", f"test_logits_epoch_{e:03d}.npy") for e in (0, 1, 2)]))
print(f"       epoch train time: {[round(row['epoch_train_seconds'], 2) for row in rows[1:]]} s, "
      f"peak memory {rows[1]['peak_gpu_memory_gb']:.2f} GB")

# 2. agreement with the earlier run (same seeds, same init => same step-0 evaluation)
if PREVIOUS_RUN_HISTORY.is_file():
    previous = json.loads(PREVIOUS_RUN_HISTORY.read_text())
    previous_by_step = {record["step"]: record for record in previous}
    check("step-0 test loss equals the earlier run (identical initialization)",
          abs(rows[0]["test_loss"] - previous_by_step[0]["test_loss"]) < 1e-3,
          f"(new {rows[0]['test_loss']:.4f} vs old {previous_by_step[0]['test_loss']:.4f})")
    # Early-epoch accuracy at lr 0.1 is chaotic: cuDNN autotuning (benchmark=True) is not deterministic across
    # processes, so the same seeds gave 0.43 / 0.48 / 0.52 at epoch 1 in repeated launches. Hence a loose tolerance.
    for epoch in (1, 2):
        old = previous_by_step[epoch * STEPS_PER_EPOCH]
        check(f"epoch {epoch} test accuracy within 0.15 of the earlier run", abs(rows[epoch]["test_accuracy"] - old["test_acc"]) < 0.15,
              f"(new {rows[epoch]['test_accuracy']:.4f} vs old {old['test_acc']:.4f})")
else:
    print("[SKIP] earlier run history not found")

# 3. subset run: keep/prune fraction must be recorded in config and mlflow
subset = torch.randperm(data.num_train_examples, generator=torch.Generator().manual_seed(0))[:10000].cuda()
subset_config = TrainingConfig(run_number=2, stop_after_epochs=1, checkpoint_epochs=(0, 1),
                               extra_parameters={"score_type": "random", "selection_rule": "check"})
subset_result = train_with_fixed_steps(data, subset_config, CHECK_DIRECTORY / "subset", train_indices=subset,
                                       mlflow_experiment="stage1-checks", mlflow_run_name="check/subset_20_percent")
saved_config = json.loads((CHECK_DIRECTORY / "subset" / "config.json").read_text())
check("config.json records keep_fraction 0.2 and prune_fraction 0.8",
      saved_config["keep_fraction"] == 0.2 and saved_config["prune_fraction"] == 0.8)
check("subset run trains the same number of steps as the full run", subset_result.epoch_rows[-1]["step"] == STEPS_PER_EPOCH)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment = mlflow.get_experiment_by_name("stage1-checks")
latest = mlflow.search_runs([experiment.experiment_id], filter_string="tags.mlflow.runName = 'check/subset_20_percent'",
                            order_by=["attributes.start_time DESC"], max_results=1)
check("mlflow run has keep_fraction / score_type parameters",
      len(latest) == 1 and latest.iloc[0]["params.keep_fraction"] == "0.2" and latest.iloc[0]["params.score_type"] == "random")
check("mlflow run finished", len(latest) == 1 and latest.iloc[0]["status"] == "FINISHED")

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}")
sys.exit(1 if failures else 0)
