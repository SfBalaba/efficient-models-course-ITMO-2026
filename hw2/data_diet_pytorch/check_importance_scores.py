"""Stage-2 check of the score code: python check_importance_scores.py

Trains one epoch, then compares every score with an independent reference:
    EL2N / L1 / margins / loss / entropy  vs  float64 numpy transcriptions of the original scores.py
    ForgettingTracker (real training steps)  vs  the ORIGINAL forgetting.py replayed on the same batches
    GraNd (torch.func.vmap)  vs  a plain per-example autograd loop
"""
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from cifar10_data import NUM_CLASSES, load_cifar10
from importance_scores import (ERROR_SCORE_NAMES, ForgettingTracker, compute_error_scores, compute_grand_scores,
                               full_fp32_precision)
from training_loop import TrainingConfig, train_with_fixed_steps

CHECK_DIRECTORY = Path(__file__).resolve().parent / "exps" / "stage2_checks"
ORIGINAL_FORGETTING_FILE = Path(__file__).resolve().parents[1] / "data_diet/data_diet/forgetting.py"
failures = []


def check(description: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {description} {detail}")
    if not condition:
        failures.append(description)


class RecordingObserver:
    """Stores what every step saw, so the original forgetting.py can be replayed afterwards."""
    def __init__(self):
        self.batches = []

    def observe_step(self, batch_indices, logits, labels):
        self.batches.append((batch_indices.cpu().numpy(), (logits.argmax(1) == labels).cpu().numpy().astype(int)))

    def summarize_epoch(self):
        return {}


data = load_cifar10()
shutil.rmtree(CHECK_DIRECTORY, ignore_errors=True)
tracker, recorder_observer = ForgettingTracker(data.num_train_examples), RecordingObserver()
result = train_with_fixed_steps(data, TrainingConfig(run_number=1, stop_after_epochs=3, checkpoint_epochs=()),
                                CHECK_DIRECTORY / "three_epochs", step_observers=[tracker, recorder_observer])
model = result.model
images, labels = data.train_images[:5000], data.train_labels[:5000]

# 1. forgetting: replay the very same batches through the original implementation
spec = importlib.util.spec_from_file_location("original_forgetting", ORIGINAL_FORGETTING_FILE)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
stats = original.init_forget_stats(SimpleNamespace(num_train_examples=data.num_train_examples))
for batch_indices, correct in recorder_observer.batches:
    stats = original.update_forget_stats(stats, batch_indices, correct)
reference_score = stats.num_forgets.copy()
reference_score[stats.never_correct] = np.inf
snapshot = tracker.snapshot()
check("forgetting scores identical to the original forgetting.py (incl. +inf for never learned)",
      np.array_equal(snapshot["forgetting_score"], reference_score))
summary = tracker.summarize_epoch()
check("tracker summary agrees with the snapshot", summary["examples_never_learned_so_far"]
      == int(np.isinf(snapshot["forgetting_score"]).sum()), str({k: int(v) for k, v in summary.items()}))
check("first_learned_step is set exactly for learned examples",
      bool(np.array_equal(snapshot["first_learned_step"] >= 0, np.isfinite(snapshot["forgetting_score"]))))
check("the real run produced forgetting events, so the comparison above is not trivial",
      summary["forgetting_events_total"] > 0, f"({int(summary['forgetting_events_total'])} events)")

# synthetic stress test: 30 passes over 1000 examples, random correctness, replayed through both implementations
generator = torch.Generator().manual_seed(0)
synthetic_tracker = ForgettingTracker(1000, device="cpu")
synthetic_stats = original.init_forget_stats(SimpleNamespace(num_train_examples=1000))
for _ in range(30):
    for batch_indices in torch.randperm(1000, generator=generator).split(50):
        correct = torch.rand(len(batch_indices), generator=generator) < 0.6
        logits = F.one_hot(torch.where(correct, 0, 1), NUM_CLASSES).float()
        synthetic_tracker.observe_step(batch_indices, logits, torch.zeros(len(batch_indices), dtype=torch.long))
        synthetic_stats = original.update_forget_stats(synthetic_stats, batch_indices.numpy(), correct.numpy().astype(int))
synthetic_reference = synthetic_stats.num_forgets.copy()
synthetic_reference[synthetic_stats.never_correct] = np.inf
check("forgetting scores identical to the original on a 30-pass synthetic sequence",
      np.array_equal(synthetic_tracker.snapshot()["forgetting_score"], synthetic_reference),
      f"(max score {synthetic_reference[np.isfinite(synthetic_reference)].max():.0f})")

# 2. error scores against float64 numpy transcriptions of the original definitions
with torch.no_grad(), full_fp32_precision():
    model.eval()
    logits = model(images.contiguous(memory_format=torch.channels_last)).double().cpu().numpy()
    model.train()
scores = compute_error_scores(model, images, labels)
true_labels = labels.cpu().numpy()
probabilities = np.exp(logits - logits.max(1, keepdims=True))
probabilities /= probabilities.sum(1, keepdims=True)
one_hot = np.eye(NUM_CLASSES)[true_labels]
is_true = one_hot.astype(bool)
wrong_minus_true = probabilities[~is_true].reshape(len(true_labels), -1) - probabilities[is_true].reshape(-1, 1)
references = {
    "el2n_score": np.linalg.norm(probabilities - one_hot, ord=2, axis=1),
    "l1_error_score": np.linalg.norm(probabilities - one_hot, ord=1, axis=1),
    "max_margin_score": wrong_minus_true.max(axis=1),
    "sum_margin_score": wrong_minus_true.sum(axis=1),
    "loss": -np.log(probabilities[is_true]),
    "entropy": -(probabilities * np.log(probabilities)).sum(axis=1),
    "true_class_probability": probabilities[is_true],
}
for name, reference in references.items():
    check(f"{name} matches the reference", np.allclose(scores[name], reference, rtol=1e-3, atol=1e-5),
          f"(max abs diff {np.abs(scores[name] - reference).max():.2e})")
check("predicted_class equals argmax of logits", np.array_equal(scores["predicted_class"], logits.argmax(1)))
check("all error-score names are produced", set(ERROR_SCORE_NAMES) == set(scores))
check("EL2N lies in [0, sqrt(2)]", scores["el2n_score"].min() >= 0 and scores["el2n_score"].max() <= np.sqrt(2) + 1e-5)

# 3. GraNd against a plain autograd loop, plus side effects
buffers_before = {name: tensor.clone() for name, tensor in model.named_buffers()}
loop_scores = []
with full_fp32_precision():
    model.eval()
    for index in range(8):
        model.zero_grad()
        F.cross_entropy(model(images[index:index + 1].contiguous(memory_format=torch.channels_last)), labels[index:index + 1]).backward()
        loop_scores.append(torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters())).item())
    model.train()
model.zero_grad()
grand = compute_grand_scores(model, images[:8], labels[:8], chunk_size=4)
check("GraNd (vmap) matches the per-example autograd loop", np.allclose(grand, loop_scores, rtol=1e-3),
      f"(max rel diff {np.max(np.abs(grand - loop_scores) / np.abs(loop_scores)):.1e})")
check("GraNd does not depend on the chunk size",
      np.allclose(compute_grand_scores(model, images[:16], labels[:16], chunk_size=3),
                  compute_grand_scores(model, images[:16], labels[:16], chunk_size=16), rtol=1e-4))
check("scoring leaves BatchNorm statistics and train mode untouched",
      model.training and all(torch.equal(buffers_before[name], tensor) for name, tensor in model.named_buffers()))
sample_grand = compute_grand_scores(model, images[:512], labels[:512])
correlation = spearmanr(sample_grand, scores["el2n_score"][:512]).statistic
check("GraNd is positive and correlates with EL2N", bool((sample_grand > 0).all()) and correlation > 0.2,
      f"(Spearman {correlation:.2f})")

shutil.rmtree(CHECK_DIRECTORY, ignore_errors=True)
print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}")
sys.exit(1 if failures else 0)
