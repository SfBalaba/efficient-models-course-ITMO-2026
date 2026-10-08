"""python run_full_data.py <ROOT> <EXP> <RUN> [--epochs 200] [--precision bf16|tf32|fp32] [--no-mlflow]
PyTorch counterpart of scripts/run_full_data.py: one independent run on the full CIFAR-10 train set,
saved to <ROOT>/exps/<EXP>/run_<RUN>."""
import argparse
import ddt

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("exp"); ap.add_argument("run", type=int)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--precision", default="bf16", choices=["bf16", "tf32", "fp32"])
ap.add_argument("--no-mlflow", action="store_true")
ap.add_argument("--mlflow-exp", default="data-diet-cifar10")
a = ap.parse_args()

data = ddt.load_cifar10()
res = ddt.train(data, a.run, f"{a.root}/exps/{a.exp}/run_{a.run}", epochs=a.epochs, precision=a.precision,
                mlflow_exp=None if a.no_mlflow else a.mlflow_exp, run_name=f"{a.exp}/full/run_{a.run}")
h = res["history"][-1]
print(f"DONE: final test acc {h['test_acc']:.4f} | train time {res['seconds'] / 60:.1f} min")
