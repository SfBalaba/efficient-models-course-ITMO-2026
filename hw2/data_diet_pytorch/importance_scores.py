"""Per-example importance scores from the Data Diet paper (arXiv 2107.07075).

    EL2N      || softmax(f(x)) - onehot(y) ||_2          one forward pass, computed early in training
    GraNd     || grad_w loss(f(x), y) ||_2               gradient w.r.t. ALL weights, one example at a time
    forgetting number of times an example goes from classified-correctly to incorrectly during training

EL2N and GraNd are computed at a checkpoint exactly as in the original scores.py: BatchNorm in eval mode, no
augmentation, full fp32 (TF32 off). Forgetting is tracked at every training step through `ForgettingTracker`, which
plugs into `training_loop` as a step observer.
"""
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from cifar10_data import NUM_CLASSES

ERROR_SCORE_NAMES = ("el2n_score", "l1_error_score", "max_margin_score", "sum_margin_score", "loss",
                     "entropy", "true_class_probability", "predicted_class")


@contextmanager
def full_fp32_precision():
    """Disable TF32 so that scores do not depend on the precision used for training."""
    previous = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous


@torch.no_grad()
def compute_error_scores(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor,
                         batch_size: int = 2000) -> dict:
    """EL2N and free side statistics from a single forward pass. Every value is a float32 array (one entry per example),
    except predicted_class (int16). margins are p_wrong - p_true as in the original `get_margin_error`."""
    was_training = model.training
    model.eval()
    collected = {name: [] for name in ERROR_SCORE_NAMES}
    with full_fp32_precision():
        for start in range(0, len(images), batch_size):
            batch_images = images[start:start + batch_size].contiguous(memory_format=torch.channels_last)
            batch_labels = labels[start:start + batch_size]
            logits = model(batch_images).float()
            log_probabilities = F.log_softmax(logits, dim=1)
            probabilities = log_probabilities.exp()
            is_true_class = F.one_hot(batch_labels, NUM_CLASSES).bool()
            error = probabilities - is_true_class.float()
            true_probability = probabilities[is_true_class]
            margins = probabilities - true_probability[:, None]
            collected["el2n_score"].append(error.norm(dim=1))
            collected["l1_error_score"].append(error.abs().sum(dim=1))
            collected["max_margin_score"].append(margins.masked_fill(is_true_class, float("-inf")).max(dim=1).values)
            collected["sum_margin_score"].append(margins.sum(dim=1))
            collected["loss"].append(-log_probabilities[is_true_class])
            collected["entropy"].append(-(probabilities * log_probabilities).sum(dim=1))
            collected["true_class_probability"].append(true_probability)
            collected["predicted_class"].append(logits.argmax(dim=1).to(torch.int16))
    model.train(was_training)
    return {name: torch.cat(parts).cpu().numpy() for name, parts in collected.items()}


def compute_grand_scores(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor,
                         chunk_size: int = 64) -> np.ndarray:
    """GraNd: L2 norm of the loss gradient w.r.t. all parameters, per example (exact, via torch.func.vmap).
    The chunk size is halved automatically when the per-example gradients do not fit into GPU memory."""
    was_training = model.training
    model.eval()
    parameters = {name: tensor.detach() for name, tensor in model.named_parameters()}
    buffers = {name: tensor.detach() for name, tensor in model.named_buffers()}

    def loss_of_one_example(parameters, buffers, image, label):
        logits = functional_call(model, (parameters, buffers), (image.unsqueeze(0),))
        return F.cross_entropy(logits, label.unsqueeze(0))

    gradients_per_example = vmap(grad(loss_of_one_example), in_dims=(None, None, 0, 0))
    scores = torch.empty(len(images), device=images.device)
    start = 0
    with full_fp32_precision():
        while start < len(images):
            end = min(start + chunk_size, len(images))
            try:
                gradients = gradients_per_example(parameters, buffers, images[start:end], labels[start:end])
                squared_norm = sum(g.flatten(1).pow(2).sum(dim=1) for g in gradients.values())
                scores[start:end] = squared_norm.sqrt()
                start = end
            except torch.cuda.OutOfMemoryError:
                if chunk_size == 1:
                    raise
                torch.cuda.empty_cache()
                chunk_size //= 2
    model.train(was_training)
    return scores.cpu().numpy()


class ForgettingTracker:
    """Forgetting events (Toneva et al. 2019), updated at every step with the batch's train-mode predictions
    (same rule as `update_forget_stats` in the original forgetting.py). Works as a `training_loop` step observer."""

    def __init__(self, num_examples: int, device: str = "cuda"):
        self.previously_correct = torch.zeros(num_examples, dtype=torch.int32, device=device)
        self.forgetting_events = torch.zeros(num_examples, dtype=torch.float32, device=device)
        self.ever_learned = torch.zeros(num_examples, dtype=torch.bool, device=device)
        self.first_learned_step = torch.full((num_examples,), -1, dtype=torch.int32, device=device)
        self.steps_observed = 0

    @torch.no_grad()
    def observe_step(self, batch_indices: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self.steps_observed += 1
        correct = (logits.argmax(dim=1) == labels).to(torch.int32)
        was_correct = self.previously_correct[batch_indices]
        self.forgetting_events[batch_indices] += (was_correct > correct).float()
        newly_learned = (correct == 1) & ~self.ever_learned[batch_indices]
        self.first_learned_step[batch_indices[newly_learned]] = self.steps_observed
        self.previously_correct[batch_indices] = correct
        self.ever_learned[batch_indices] |= correct.bool()

    @torch.no_grad()
    def summarize_epoch(self) -> dict:
        seen = self.ever_learned | (self.forgetting_events > 0)
        return {
            "forgetting_events_total": self.forgetting_events.sum().item(),
            "examples_forgotten_at_least_once": (self.forgetting_events > 0).sum().item(),
            "examples_never_learned_so_far": (~self.ever_learned).sum().item(),
            "examples_unforgettable_so_far": (self.ever_learned & (self.forgetting_events == 0)).sum().item(),
            "examples_learned_so_far": seen.sum().item(),
        }

    @torch.no_grad()
    def snapshot(self) -> dict:
        """Forgetting score as in the paper: event count, +inf for examples that were never classified correctly."""
        score = self.forgetting_events.clone()
        score[~self.ever_learned] = float("inf")
        return {"forgetting_score": score.cpu().numpy(),
                "forgetting_events": self.forgetting_events.cpu().numpy(),
                "first_learned_step": self.first_learned_step.cpu().numpy()}
