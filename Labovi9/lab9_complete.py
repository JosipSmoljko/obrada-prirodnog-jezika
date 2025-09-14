#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab 9 — Metode optimizacije
Kompletan kod: treniranje Random Forest modela za predikciju cijene stana i
usporedba metoda optimizacije hiperparametara:
- Grid Search
- Halving Grid Search
- Random Search
- Halving Random Search
- Bayesian Optimization (BayesSearchCV iz scikit-optimize)

Napomena: Za Bayesian Optimization potrebno je: pip install scikit-optimize
"""

import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Halving pretrage su u eksperimentalnom modulu
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

# Pokušaj uvoza BayesSearchCV (možda nije instaliran)
try:
    from skopt import BayesSearchCV
    HAVE_SKOPT = True
except Exception as e:
    HAVE_SKOPT = False
    _SKOPT_IMPORT_ERR = e

RANDOM_STATE = 42

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def split(df: pd.DataFrame):
    X = df.drop('Cijena', axis=1)
    y = df['Cijena']
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

def evaluate(name: str, search, X_val, y_val):
    best = search.best_estimator_
    y_pred = best.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    return {
        "method": name,
        "best_params": search.best_params_,
        "best_cv_score": getattr(search, "best_score_", np.nan),
        "val_mse": mse,
        "val_r2": r2
    }

def main(csv_path="lab9_dataset.csv", out_results_csv="lab9_results.csv"):
    df = load_data(csv_path)
    X_train, X_val, y_train, y_val = split(df)

    base_model = RandomForestRegressor(random_state=RANDOM_STATE)

    # Prostor hiperparametara
    grid = {
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }

    dist = {
        "n_estimators": np.arange(50, 601, 25),
        "max_depth": [None] + list(np.arange(5, 41, 5)),
        "min_samples_split": np.arange(2, 21),
        "min_samples_leaf": np.arange(1, 11),
        "max_features": ["sqrt", "log2", None]
    }

    # K-fold radi stabilnije procjene na malom skupu
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    timings = []

    def run_and_time(name, search):
        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        t1 = time.perf_counter()
        timings.append({"method": name, "seconds": t1 - t0})
        results.append(evaluate(name, search, X_val, y_val))

    # 1) Grid Search
    gs = GridSearchCV(
        estimator=base_model,
        param_grid=grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    run_and_time("GridSearchCV", gs)

    # 2) Halving Grid Search
    hgs = HalvingGridSearchCV(
        estimator=base_model,
        param_grid=grid,
        factor=2,
        resource="n_estimators",
        max_resources=600,  # povećavamo šumu kako resurs
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    run_and_time("HalvingGridSearchCV", hgs)

    # 3) Randomized Search
    rs = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=dist,
        n_iter=40,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    run_and_time("RandomizedSearchCV", rs)

    # 4) Halving Random Search
    hrs = HalvingRandomSearchCV(
        estimator=base_model,
        param_distributions=dist,
        factor=2,
        resource="n_estimators",
        max_resources=600,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    run_and_time("HalvingRandomSearchCV", hrs)

    # 5) Bayesian Optimization (ako je dostupno)
    if HAVE_SKOPT:
        # Pretvori u skopt prostore
        from skopt.space import Integer, Categorical
        search_spaces = {
            "n_estimators": Integer(50, 600),
            "max_depth": Categorical([None] + list(range(5, 41, 5))),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 10),
            "max_features": Categorical(["sqrt", "log2", None])
        }
        bs = BayesSearchCV(
            estimator=base_model,
            search_spaces=search_spaces,
            n_iter=40,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )
        run_and_time("BayesSearchCV", bs)
    else:
        warnings.warn(
            "skopt nije instaliran; preskačem BayesSearchCV. "
            f"Detalji greške: {_SKOPT_IMPORT_ERR}"
        )

    # Sažeci
    results_df = pd.DataFrame(results)
    timings_df = pd.DataFrame(timings)

    # Spoji za lakšu usporedbu
    summary_df = results_df.merge(timings_df, on="method")
    print("\\n--- REZULTATI ---")
    print(summary_df)

    # Spremi CSV
    results_path = out_results_csv
    summary_df.to_csv(results_path, index=False)
    print(f"Rezultati zapisani u: {results_path}")

if __name__ == "__main__":
    main()
