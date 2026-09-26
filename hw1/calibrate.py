"""Калибровка theta для latency() и energy() из equations.py.

Фит — на базовой сетке (is_validation == 0), метрики — на валидационных точках.
Предсказания для метрик считаются через latency() / energy() из equations.py.

Запуск:
    python calibrate.py   # -> results/theta.json
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

import numpy as np
from equations import energy, flops, latency, memory


HW_DIR = Path(__file__).resolve().parent
RESULTS_DIR = HW_DIR / "results"


def load_split(csv_path=RESULTS_DIR / "measurements.csv"):
    df = pd.read_csv(csv_path)
    ok = df[df.status == "OK"]
    return ok[ok.is_validation == 0], ok[ok.is_validation == 1]


def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"metrics": {"mae": mae, "mse": mse, "r2": r2, "mape": mape}}


def features(df):
    S, B = df.S.to_numpy(), df.B.to_numpy()
    return np.vstack([flops(S, B), memory(S, B)]).T


def main():
    train, val = load_split()
    print(f"train: {len(train)} точек, validation: {len(val)} точек")

    model_latency = LinearRegression(fit_intercept=True).fit(features(train), train.latency_s)
    theta_latency = {
        "theta_launch": model_latency.intercept_,
        "theta_comp": model_latency.coef_[0],
        "theta_mem": model_latency.coef_[1],
    }
    latency_pred_val = latency(val.S.to_numpy(), val.B.to_numpy(), theta_latency)
    print("latency model metrics: ", eval_metrics(val.latency_s, latency_pred_val))

    latency_pred_train = latency(train.S.to_numpy(), train.B.to_numpy(), theta_latency).reshape(-1, 1)
    model_energy = LinearRegression(fit_intercept=True).fit(latency_pred_train, train.energy_j)
    theta_energy = {
        "theta_launch": model_energy.intercept_,
        "theta_power": model_energy.coef_[0],
        "latency": theta_latency,
    }
    energy_pred_val = energy(val.S.to_numpy(), val.B.to_numpy(), theta_energy)
    print("energy model metrics: ", eval_metrics(val.energy_j, energy_pred_val))

    theta = {"latency": theta_latency, "energy": theta_energy}
    (RESULTS_DIR / "theta.json").write_text(json.dumps(theta, indent=2))


if __name__ == "__main__":
    main()
