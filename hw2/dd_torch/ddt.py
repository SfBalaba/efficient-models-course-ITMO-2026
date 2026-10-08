"""PyTorch port of mansheej/data_diet (arXiv 2107.07075): CIFAR-10 + ResNet18-lowres, fixed-step SGD training."""

import json, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.datasets import CIFAR10

DATA_ROOT = os.environ.get("CIFAR_ROOT", "/home/sofya/Downloads/data")
MLFLOW_URI = "sqlite:////home/sofya/Documents/gitlab/efficent_net/hw2/mlflow.db"
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
EP_STEPS = 390  # 50000 // 128
META_MODEL_SEED, META_TRAIN_SEED, SEED_INCR = (
    42,
    4242,
    424242,
)  # as in scripts/run_full_data.py


def load_cifar10(device="cuda"):
    out = {}
    for split, train in (("train", True), ("test", False)):
        ds = CIFAR10(DATA_ROOT, train=train, download=False)
        x = (ds.data.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
        out[split] = (x, torch.tensor(ds.targets, dtype=torch.long, device=device))
    return out


def augment(x, rng, gen):
    """Official pipeline: REFLECT pad 4, ONE random 32x32 crop offset shared by the whole batch
    (tf.image.random_crop on a [B,H,W,C] tensor), independent horizontal flip per image.
    """
    p = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = rng.randint(0, 9, size=2)
    x = p[:, :, i : i + 32, j : j + 32]
    flip = torch.rand(x.shape[0], device=x.device, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.proj = None
        if stride != 1 or cin != cout:
            self.proj = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        y = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu((x if self.proj is None else self.proj(x)) + y)


class ResNet18(nn.Module):
    """ResNet18-v1 low-res: 3x3 stride-1 stem, no max-pool; flax-style lecun_normal init."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()
        )
        blocks, cin = [], 64
        for cout, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            blocks += [Block(cin, cout, stride), Block(cout, cout, 1)]
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                std = math.sqrt(1.0 / m.weight[0].numel()) / 0.87962566103423978
                nn.init.trunc_normal_(m.weight, 0.0, std, -2 * std, 2 * std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.blocks(self.stem(x)), 1).flatten(1))


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    loss = correct = 0.0
    for k in range(0, len(x), bs):
        out = model(x[k : k + bs].contiguous(memory_format=torch.channels_last)).float()
        loss += F.cross_entropy(out, y[k : k + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[k : k + bs]).sum().item()
    model.train()
    return loss / len(x), correct / len(x)


@torch.no_grad()
def el2n_scores(model, x, y, num_classes=10, bs=2000):
    """EL2N = || softmax(f(x)) - onehot(y) ||_2, eval-mode BN, no augmentation, fp32 (as scores.py)."""
    model.eval()
    out = []
    for k in range(0, len(x), bs):
        p = F.softmax(
            model(x[k : k + bs].contiguous(memory_format=torch.channels_last)).float(),
            dim=1,
        )
        out.append((p - F.one_hot(y[k : k + bs], num_classes).float()).norm(dim=1))
    model.train()
    return torch.cat(out).cpu().numpy()


def train(
    data,
    run,
    save_dir,
    idx=None,
    epochs=200,
    bs=128,
    precision="bf16",
    save_epochs=(0, 4, 8, 10, 12, 16, 20, 100, 200),
    track_forgetting=True,
    mlflow_exp=None,
    run_name=None,
    extra_params=None,
):
    """Fixed-step SGD (nesterov, lr 0.1, wd 5e-4, /5 at 30/60/80% of training). idx = training subset (None = all).
    Number of steps does NOT depend on the subset size (as in the paper). Returns dict with history and final model.
    """
    dev = "cuda"
    total_steps = epochs * EP_STEPS
    model_seed, train_seed = (
        META_MODEL_SEED + run * SEED_INCR,
        META_TRAIN_SEED + run * SEED_INCR,
    )
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = precision != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = precision != "fp32"
    os.makedirs(save_dir + "/ckpts", exist_ok=True)
    if track_forgetting:
        os.makedirs(save_dir + "/forget_scores", exist_ok=True)

    xtr, ytr = data["train"]
    xte, yte = data["test"]
    N = len(xtr)
    idx = torch.arange(N, device=dev) if idx is None else idx.to(dev)
    torch.manual_seed(model_seed)
    model = ResNet18().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4
    )
    g = torch.Generator(device=dev).manual_seed(train_seed)
    np_rng = np.random.RandomState(train_seed)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16")
    bounds = [int(f * total_steps) for f in (0.3, 0.6, 0.8)]
    save_steps = {int(e * EP_STEPS) for e in save_epochs if e <= epochs}

    prev_acc = torch.zeros(N, dtype=torch.int32, device=dev)
    num_forgets = torch.zeros(N, dtype=torch.float32, device=dev)
    ever_correct = torch.zeros(N, dtype=torch.bool, device=dev)

    cfg = dict(
        run=run,
        epochs=epochs,
        total_steps=total_steps,
        batch_size=bs,
        precision=precision,
        lr=0.1,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
        decay_factor=0.2,
        decay_steps=bounds,
        model_seed=model_seed,
        train_seed=train_seed,
        n_train=len(idx),
        track_forgetting=track_forgetting,
        torch=torch.__version__,
        augment="reflect-pad4/batch-shared-crop/per-image-flip",
        **(extra_params or {}),
    )
    json.dump(cfg, open(save_dir + "/args.json", "w"), indent=1)

    ml = None
    if mlflow_exp:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(mlflow_exp)
        ml = mlflow
        ml.start_run(run_name=run_name or os.path.basename(save_dir.rstrip("/")))
        ml.log_params(cfg)

    def ckpt(step):
        torch.save(model.state_dict(), f"{save_dir}/ckpts/ckpt_{step}.pt")
        if track_forgetting:
            f = num_forgets.clone()
            f[~ever_correct] = float("inf")
            np.save(f"{save_dir}/forget_scores/ckpt_{step}.npy", f.cpu().numpy())

    history, step, ptr, perm = [], 0, 0, None
    test_loss, test_acc = evaluate(model, xte, yte)
    history.append(dict(step=0, test_acc=test_acc, test_loss=test_loss))
    if 0 in save_steps:
        ckpt(0)
    if ml:
        ml.log_metrics(dict(test_acc=test_acc, test_loss=test_loss), step=0)
    torch.cuda.synchronize()
    t0 = t_ep = time.perf_counter()
    loss_sum = torch.zeros((), device=dev)
    acc_sum = torch.zeros((), device=dev)
    n_b = 0
    n = len(idx)
    model.train()
    while step < total_steps:
        if (
            perm is None or ptr + bs > n
        ):  # reshuffle at end of pass, drop the incomplete tail (as official)
            perm, ptr = idx[torch.randperm(n, device=dev, generator=g)], 0
        b = perm[ptr : ptr + bs]
        ptr += bs
        step += 1
        x = augment(xtr[b], np_rng, g).contiguous(memory_format=torch.channels_last)
        y = ytr[b]
        lr = 0.1 * 0.2 ** sum(step >= s for s in bounds)
        for pg in opt.param_groups:
            pg["lr"] = lr
        with amp:
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            c = logits.argmax(1) == y
            loss_sum += loss.detach()
            acc_sum += c.float().mean()
            n_b += 1
            if track_forgetting:
                ci = c.to(torch.int32)
                num_forgets[b] += (prev_acc[b] > ci).float()
                prev_acc[b] = ci
                ever_correct[b] |= c
        if step % EP_STEPS == 0 or step == total_steps:
            torch.cuda.synchronize()
            t_train = time.perf_counter() - t_ep
            test_loss, test_acc = evaluate(model, xte, yte)
            rec = dict(
                step=step,
                epoch=step / EP_STEPS,
                lr=lr,
                train_loss=(loss_sum / n_b).item(),
                train_acc=(acc_sum / n_b).item(),
                test_acc=test_acc,
                test_loss=test_loss,
                epoch_time_s=t_train,
                elapsed_min=(time.perf_counter() - t0) / 60,
            )
            history.append(rec)
            if ml:
                ml.log_metrics({k: v for k, v in rec.items() if k != "step"}, step=step)
            print(
                f"{step / total_steps * 100:6.2f}% | {rec['elapsed_min']:5.1f}m | step {step:6d} | lr {lr:.4f} | "
                f"train loss {rec['train_loss']:.3f} acc {rec['train_acc']:.3f} | test acc {test_acc:.4f}",
                flush=True,
            )
            loss_sum.zero_()
            acc_sum.zero_()
            n_b = 0
            if step in save_steps or step == total_steps:
                ckpt(step)
            torch.cuda.synchronize()
            t_ep = time.perf_counter()
    total = time.perf_counter() - t0
    json.dump(history, open(save_dir + "/history.json", "w"))
    if ml:
        ml.log_metrics(
            dict(final_test_acc=history[-1]["test_acc"], total_train_time_s=total)
        )
        ml.end_run()
    return dict(model=model, history=history, seconds=total)
