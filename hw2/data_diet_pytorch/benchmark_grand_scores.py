"""Measure how long one GraNd (and one EL2N) pass over the 50 000 training examples takes: python benchmark_grand_scores.py"""
import time

import torch

from cifar10_data import NUM_CLASSES, load_cifar10
from importance_scores import compute_error_scores, compute_grand_scores
from resnet18_lowres import ResNet18LowRes

SAMPLE_SIZE = 1024
torch.backends.cudnn.benchmark = True  # same setting as the training loop
data = load_cifar10()
model = ResNet18LowRes(NUM_CLASSES).cuda().to(memory_format=torch.channels_last)
images, labels = data.train_images[:SAMPLE_SIZE], data.train_labels[:SAMPLE_SIZE]


def seconds_per_full_pass(function, repeats=2):
    function()  # warm-up (cuDNN autotune, allocator)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    per_example = (time.perf_counter() - start) / repeats / SAMPLE_SIZE
    return per_example * data.num_train_examples, torch.cuda.max_memory_allocated() / 2**30


seconds, memory = seconds_per_full_pass(lambda: compute_error_scores(model, images, labels))
print(f"EL2N + side statistics: {seconds:6.2f} s per 50k examples, peak {memory:.2f} GB")
for chunk_size in (16, 32, 64, 128):
    try:
        seconds, memory = seconds_per_full_pass(lambda: compute_grand_scores(model, images, labels, chunk_size))
        print(f"GraNd chunk {chunk_size:3d}:        {seconds:6.1f} s per 50k examples, peak {memory:.2f} GB")
    except torch.cuda.OutOfMemoryError:
        print(f"GraNd chunk {chunk_size:3d}:        out of memory")
        torch.cuda.empty_cache()
