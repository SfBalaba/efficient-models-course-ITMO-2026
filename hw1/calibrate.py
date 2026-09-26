"""Калибровка theta для latency() и energy() из equations.py.

Заготовка: загрузка замеров и разбиение на train (базовая сетка) / validation.
Сам фит зависит от формул в equations.py — TODO.

Запуск:
    python calibrate.py   # -> results/theta.json
"""

import json
from pathlib import Path

import pandas as pd

from pandas import test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from equations import flops, memory
import numpy as np


HW_DIR = Path(__file__).resolve().parent
RESULTS_DIR = HW_DIR / "results"


def load_split(csv_path=RESULTS_DIR / "measurements.csv"):
    df = pd.read_csv(csv_path)
    ok = df[df.status == "OK"]
    return ok[ok.is_validation == 0], ok[ok.is_validation == 1]


def eval_model(model, X, y_true):
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"metrics": {"mae": mae, "mse": mse, "r2": r2, "mape": mape}}


def main():
    train, val = load_split()
    print(f"train: {len(train)} точек, validation: {len(val)} точек")
    
    
    train['analytical_flops'] = train.apply(lambda row: flops(row.S, row.B), axis=1) 
    val['analytical_flops'] = val.apply(lambda row: flops(row.S, row.B), axis=1)  
       
    train['analytical_mem'] = train.apply(lambda row: memory(row.S, row.B), axis=1) 
    val['analytical_mem'] = val.apply(lambda row: memory(row.S, row.B), axis=1)     
    
    X_train = np.vstack([train['analytical_flops']  , train['analytical_mem']]).T
    y_train = train['latency_s']   
    
    X_test = np.vstack([val['analytical_flops'] , val['analytical_mem']]).T
    y_test_latency = val['latency_s']

    model_latency = LinearRegression(fit_intercept=True).fit(X_train, y_train)
    metrics_latency = eval_model(model_latency, X_test, y_test_latency)
    print("latency model metrics: ", metrics_latency)
    
    latency_pred_train = model_latency.predict(X_train).reshape(-1, 1)
    y_true_energy_train = train['energy_j']
    
    
    y_pred_latency_test = model_latency.predict(X_test).reshape(-1, 1)
    y_true_energy_test = val['energy_j']
    
    
    model_energy = LinearRegression(fit_intercept=True).fit(latency_pred_train, y_true_energy_train)
    metrics_energy = eval_model(model_energy, y_pred_latency_test, y_true_energy_test)
    print("energy model metrics: ", metrics_energy)
    
    
    theta = {
        "latency": {
            "theta_launch": model_latency.intercept_,
            "theta_comp": model_latency.coef_[0],
            "theta_mem": model_latency.coef_[1],
        },
        "energy": {
            "theta_launch": model_energy.intercept_,
            "theta_comp": model_energy.coef_[0],
        },
    }


    (RESULTS_DIR / "theta.json").write_text(json.dumps(theta, indent=2))


if __name__ == "__main__":
    main()
