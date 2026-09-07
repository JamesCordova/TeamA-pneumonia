#!/usr/bin/env python
r"""
Proyección a futuro (Recomendación 2 — Gopi).

Para cada experimento del manifest: entrena el modelo con TODO el histórico
disponible y pronostica las próximas FUTURE_WEEKS semanas *más allá* del último
dato real (no es backtest: no hay valor observado que comparar). Guarda el
resultado en pruebasg/results/{depto}-{etiqueta}/future.csv, que el dashboard
dibuja como una extensión punteada de la línea histórica.

Uso:
    python pruebasg/future_forecast.py
    python pruebasg/future_forecast.py --weeks 8
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silenciar logs de TensorFlow
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

from _common import ROOT, RESULTS_DIR, load_manifest, slug

sys.path.insert(0, str(ROOT))
from pneumonia.models.utils import get_departmental_data  # noqa: E402

FUTURE_WEEKS = 4

# Mapea la etiqueta del experimento a (clase, kwargs). Espeja run_grid.MODELS.
XGB_TUNED = {"n_estimators": 500, "learning_rate": 0.01,
             "max_depth": 3, "colsample_bytree": 0.7}
RF_LAGS = [1, 2, 4, 8, 26, 52]
RF_WINDOWS = [4, 13, 26, 52]


def build_model(label: str, department: str, age_group: str):
    from pneumonia.models.baselines.seasonal_naive import SeasonalNaiveForecaster
    from pneumonia.models.ml.random_forest import RandomForestModel
    from pneumonia.models.ml.xgboost import XGBoostModel
    from pneumonia.models.sarima.model import SARIMAModel
    from pneumonia.models.rnn.lstm import LSTMModel
    from pneumonia.models.rnn.gru import GRUModel

    d, a = department, age_group

    # Configs "(grid-search)": cargar mejores parámetros desde la caché
    if label.endswith("(grid-search)"):
        from _common import RESULTS_DIR
        cache = RESULTS_DIR / "best_params" / f"{slug(d)}__{label.split(' ')[0].lower()}.json"
        best = __import__("json").loads(cache.read_text(encoding="utf-8"))["best_params"]
        cls = RandomForestModel if label.startswith("RandomForest") else XGBoostModel
        return cls(d, a, **best)

    builders = {
        "SeasonalNaive (baseline)": lambda: SeasonalNaiveForecaster(d, a),
        "RandomForest (default)": lambda: RandomForestModel(d, a),
        "RandomForest (lags+windows)": lambda: RandomForestModel(
            d, a, lags=RF_LAGS, windows=RF_WINDOWS),
        "XGBoost (tuned)": lambda: XGBoostModel(d, a, xgb_params=XGB_TUNED),
        "SARIMA (Fourier K=10)": lambda: SARIMAModel(d, a, n_fourier_terms=10),
        "LSTM": lambda: LSTMModel(d, a, epochs=40),
        "GRU": lambda: GRUModel(d, a, epochs=40),
    }
    if label not in builders:
        raise KeyError(f"Sin builder para la etiqueta '{label}'")
    return builders[label]()


def forecast_one(entry: dict, weeks: int) -> bool:
    dept, age, label = entry["department"], entry["age_group"], entry["label"]
    series = get_departmental_data(dept, age_group=age)

    model = build_model(label, dept, age)
    model.fit(series)
    preds = model.predict(series, steps=weeks)

    step = series.index[-1] - series.index[-2]
    last_date = series.index[-1]
    dates = [last_date + step * (i + 1) for i in range(weeks)]

    out = pd.DataFrame({"date": dates, "predicted": preds[:weeks]})
    dest = RESULTS_DIR / slug(f"{dept}-{label}") / "future.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"  [OK] {dept} / {label}: {weeks} sem futuras -> {dest.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weeks", type=int, default=FUTURE_WEEKS,
                        help=f"Semanas a proyectar (default: {FUTURE_WEEKS})")
    args = parser.parse_args()

    manifest = [e for e in load_manifest() if e.get("ok")]
    if not manifest:
        print("No hay experimentos OK en el manifest. Corre run_grid.py primero.")
        return 1

    print(f"Proyectando {args.weeks} semanas para {len(manifest)} experimentos...")
    ok = 0
    for entry in manifest:
        try:
            ok += forecast_one(entry, args.weeks)
        except Exception as exc:
            print(f"  [FALLO] {entry['department']} / {entry['label']}: {exc}")
    print(f"\nListo: {ok}/{len(manifest)} proyecciones futuras generadas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
