#!/usr/bin/env python
r"""
Pronóstico fuera de muestra: las cuatro semanas siguientes al último dato.

La validación walk-forward responde cuánto erró cada familia en el pasado; esto
responde qué dice hoy el modelo que el criterio recomienda. Son dos cosas
distintas y el tablero las muestra por separado: la primera sostiene la
recomendación, la segunda es lo que un servicio de vigilancia usaría.

El ajuste final usa la ventana terminal de `TRAIN_SIZE` semanas, la misma
longitud de entrenamiento de cada origen del walk-forward (Etapa 6). No se
reentrena sobre el histórico completo porque entonces el modelo que pronostica
no sería el modelo que se evaluó.

Salida:
    pruebasg/results/future/futuro_<grupo>.csv
        departamento, modelo, fecha, prediccion, paso

Uso:
    python pruebasg/build_future.py
    python pruebasg/build_future.py --age_group 60plus --modelos XGBoost
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PRUEBAS_DIR = Path(__file__).resolve().parent
ROOT = PRUEBAS_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PRUEBAS_DIR))

import logging

import pandas as pd

from _common import RESULTS_DIR

logging.disable(logging.WARNING)

OUT_DIR = RESULTS_DIR / "future"
TRAIN_SIZE = 520          # igual que --train_size de la grilla
SEMANAS = 4               # igual que --horizon
MODELOS = ["SeasonalNaive", "SARIMA", "RandomForest", "XGBoost", "LSTM"]


def construir(modelo: str, departamento: str, age_group: str):
    """Instancia el modelo con la MISMA configuración de run_grid_selector."""
    d, a = departamento, age_group
    if modelo == "SeasonalNaive":
        from pneumonia.models.baselines.seasonal_naive import SeasonalNaiveForecaster
        return SeasonalNaiveForecaster(department=d, age_group=a)
    if modelo == "SARIMA":
        from pneumonia.models.sarima.model import SARIMAModel
        return SARIMAModel(department=d, age_group=a, n_fourier_terms=10)
    if modelo == "RandomForest":
        from pneumonia.models.ml.random_forest import RandomForestModel
        return RandomForestModel(department=d, age_group=a,
                                 lags=[1, 2, 4, 8, 26, 52],
                                 windows=[4, 13, 26, 52])
    if modelo == "XGBoost":
        from pneumonia.models.ml.xgboost import XGBoostModel
        return XGBoostModel(department=d, age_group=a,
                            xgb_params={"n_estimators": 500, "learning_rate": 0.01,
                                        "max_depth": 3, "colsample_bytree": 0.7})
    if modelo == "LSTM":
        from pneumonia.models.rnn.lstm import LSTMModel
        return LSTMModel(department=d, age_group=a, epochs=40)
    raise ValueError(modelo)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--age_group", default="under5", choices=["under5", "60plus"])
    p.add_argument("--departments", nargs="+", default=None)
    p.add_argument("--modelos", nargs="+", default=MODELOS)
    p.add_argument("--weeks", type=int, default=SEMANAS)
    args = p.parse_args()

    from pneumonia.models.utils import get_departmental_data
    from pneumonia.models.utils import load_weekly_cases

    if args.departments:
        departamentos = [d.upper() for d in args.departments]
    else:
        datos = load_weekly_cases(age_group=args.age_group)
        departamentos = sorted(datos["department"].unique())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filas, fallos = [], []
    t0 = time.time()

    for dept in departamentos:
        serie_completa = get_departmental_data(dept, age_group=args.age_group)
        serie = serie_completa.iloc[-TRAIN_SIZE:] if len(serie_completa) > TRAIN_SIZE \
            else serie_completa
        paso = serie.index[-1] - serie.index[-2]
        fechas = [serie.index[-1] + paso * (i + 1) for i in range(args.weeks)]

        for modelo in args.modelos:
            marca = time.time()
            try:
                obj = construir(modelo, dept, args.age_group)
                obj.fit(serie)
                yhat = obj.predict(serie, steps=args.weeks)
            except Exception as exc:  # noqa: BLE001
                fallos.append((dept, modelo, str(exc)[:110]))
                print(f"  {dept:14s} {modelo:14s} FALLO: {str(exc)[:80]}")
                continue
            for i, fecha in enumerate(fechas):
                filas.append({"departamento": dept, "modelo": modelo,
                              "fecha": fecha.date().isoformat(),
                              "prediccion": round(float(max(0.0, yhat[i])), 3),
                              "paso": i + 1})
            print(f"  {dept:14s} {modelo:14s} ok  ({time.time() - marca:.1f} s)")

    if not filas:
        print("Sin resultados.")
        return 1

    df = pd.DataFrame(filas)
    destino = OUT_DIR / f"futuro_{args.age_group}.csv"
    df.to_csv(destino, index=False, encoding="utf-8")
    print(f"\n{len(df)} filas · {df['departamento'].nunique()} departamentos "
          f"x {df['modelo'].nunique()} modelos · {time.time() - t0:.0f} s")
    print(f"-> {destino.relative_to(ROOT)}")
    if fallos:
        print(f"\n{len(fallos)} fallos:")
        for d, m, e in fallos[:12]:
            print(f"  {d} / {m}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
