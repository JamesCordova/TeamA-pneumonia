#!/usr/bin/env python
r"""
Grid search con caché de mejores parámetros (Recomendación 1 — Gopi).

Para XGBoost y RandomForest, en cada departamento:
  1. Prueba muchas combinaciones de hiperparámetros puntuándolas con un
     walk-forward LIGERO (rápido) → elige la de menor MAE promedio.
  2. Cachea la mejor en pruebasg/results/best_params/{depto}__{modelo}.json
     (así las corridas siguientes cargan los mejores parámetros al instante).
  3. Corre un walk-forward COMPLETO con esa mejor config y la guarda como un
     experimento nuevo "{Modelo} (grid-search)", que aparece en el dashboard
     junto a la versión de parámetros fijos (comparación antes/después).
  4. Genera su proyección futura.

Uso:
    python pruebasg/grid_search.py
    python pruebasg/grid_search.py --search-only   # solo busca y cachea, sin correr el WF completo

Nota: re-ejecutar run_grid.py NO borra las entradas grid-search del manifest
(este script las fusiona). Los mejores parámetros quedan en results/best_params/.
"""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import subprocess
import sys
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

from _common import RESULTS_DIR, ROOT, load_json, load_manifest, save_json, slug

sys.path.insert(0, str(ROOT))
from pneumonia.evaluation.walkforward import WalkForwardValidator  # noqa: E402
from pneumonia.models.ml.random_forest import RandomForestModel  # noqa: E402
from pneumonia.models.ml.xgboost import XGBoostModel  # noqa: E402
from pneumonia.models.utils import get_departmental_data  # noqa: E402

BEST_DIR = RESULTS_DIR / "best_params"
DEPARTMENTS = ["AMAZONAS", "LIMA"]
AGE_GROUP = "under5"

# Walk-forward de COBERTURA COMPLETA para puntuar cada combinación: step=4
# puntúa todos los orígenes (igual que la evaluación final); refit_every=13
# reentrena periódicamente para que el barrido siga siendo tratable en tiempo.
SCORE_WF = dict(initial_train_size=520, horizon=4, step=4,
                refit_every=13, window_type="sliding", min_train_size=104)

# Feature sets para RandomForest (rezagos / ventanas)
RF_FEATURES = {
    "default": {"lags": [1, 2, 4, 8, 13], "windows": [4, 13, 26]},
    "long":    {"lags": [1, 2, 4, 8, 26, 52], "windows": [4, 13, 26, 52]},
}


def rf_configs():
    for n, d, feat in itertools.product([100, 300], [8, 12, 20], RF_FEATURES):
        yield {"rf_params": {"n_estimators": n, "max_depth": d},
               "lags": RF_FEATURES[feat]["lags"], "windows": RF_FEATURES[feat]["windows"]}


def xgb_configs():
    for n, d, lr in itertools.product([300, 500], [3, 4, 6], [0.01, 0.05, 0.1]):
        yield {"xgb_params": {"n_estimators": n, "max_depth": d,
                              "learning_rate": lr, "colsample_bytree": 0.8}}


MODELS = {
    "RandomForest": (RandomForestModel, rf_configs),
    "XGBoost":      (XGBoostModel, xgb_configs),
}


# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------

def score_config(model_class, dept, cfg, series) -> float:
    """MAE promedio (h=1..4) de un walk-forward ligero. inf si falla."""
    params = {"department": dept, "age_group": AGE_GROUP, **cfg}
    try:
        wf = WalkForwardValidator(model_class=model_class, model_params=params, **SCORE_WF)
        res = wf.run(series)
        maes = [m.get("mae") for m in res["metrics_by_horizon"].values()
                if m and m.get("mae") is not None]
        return sum(maes) / len(maes) if maes else float("inf")
    except Exception:
        return float("inf")


def to_cli(model: str, cfg: dict) -> list[str]:
    """Convierte una config a flags de run_walkforward.py."""
    args = []
    if model == "RandomForest":
        p = cfg["rf_params"]
        args += ["--n_estimators", str(p["n_estimators"]), "--max_depth", str(p["max_depth"])]
        args += ["--lags", *map(str, cfg["lags"]), "--windows", *map(str, cfg["windows"])]
    else:
        p = cfg["xgb_params"]
        args += ["--n_estimators", str(p["n_estimators"]), "--max_depth", str(p["max_depth"]),
                 "--learning_rate", str(p["learning_rate"]),
                 "--colsample_bytree", str(p["colsample_bytree"])]
    return args


def search(dept: str, model: str) -> dict:
    model_class, cfg_gen = MODELS[model]
    series = get_departmental_data(dept, age_group=AGE_GROUP)
    configs = list(cfg_gen())
    print(f"\n[{dept} / {model}] probando {len(configs)} combinaciones...")

    scored = []
    for i, cfg in enumerate(configs, 1):
        s = score_config(model_class, dept, cfg, series)
        scored.append((s, cfg))
        if i % 5 == 0 or i == len(configs):
            print(f"    {i}/{len(configs)}  mejor MAE hasta ahora: "
                  f"{min(x[0] for x in scored):.3f}")

    scored.sort(key=lambda t: t[0])
    best_score, best_cfg = scored[0]

    BEST_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "department": dept, "model": model,
        "best_params": best_cfg, "best_mae_mean": round(best_score, 4),
        "best_cli": " ".join(to_cli(model, best_cfg)),
        "search": {"n_configs": len(configs), "wf_score_config": SCORE_WF},
        "top5": [{"mae_mean": round(s, 4), "params": c} for s, c in scored[:5]],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out = BEST_DIR / f"{slug(dept)}__{model.lower()}.json"
    save_json(out, payload)
    print(f"  [OK] mejor MAE={best_score:.3f} -> {out.name}")
    print(f"       best: {payload['best_cli']}")
    return payload


# ---------------------------------------------------------------------------
# Corrida completa de la mejor config + snapshot + manifest + futuro
# ---------------------------------------------------------------------------

def full_run(dept: str, model: str, best_cli: str) -> dict | None:
    label = f"{model} (grid-search)"
    cmd = [sys.executable, str(ROOT / "scripts" / "run_walkforward.py"),
           "--department", dept, "--model", model, "--age_group", AGE_GROUP,
           "--horizon", "4", "--step", "4", *best_cli.split()]
    shown = "python scripts/run_walkforward.py " + " ".join(cmd[2:])
    print(f"\n[{label}] {dept}\n  $ {shown}")

    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        print("  [FALLO]", "\n".join(r.stdout.splitlines()[-6:]))
        return None

    metrics_src = ROOT / "reports" / dept.upper() / AGE_GROUP / f"{model.lower()}_walkforward_metrics.json"
    preds_src = ROOT / "reports" / dept.upper() / AGE_GROUP / f"{model}_predictions.csv"
    snap = RESULTS_DIR / slug(f"{dept}-{label}")
    snap.mkdir(parents=True, exist_ok=True)
    config = {}
    if metrics_src.exists():
        shutil.copy2(metrics_src, snap / "metrics.json")
        config = load_json(snap / "metrics.json").get("config", {})
    if preds_src.exists():
        shutil.copy2(preds_src, snap / "predictions.csv")

    # Proyección futura para esta config
    _future_for_best(dept, model, best_cli, snap)

    print(f"  [OK] snapshot -> {snap.name}")
    return {
        "label": label, "department": dept.upper(), "model": model,
        "age_group": AGE_GROUP, "command": shown, "returncode": 0, "ok": True,
        "config": config,
        "metrics_file": str((snap / "metrics.json").relative_to(RESULTS_DIR)) if metrics_src.exists() else None,
        "pred_file": str((snap / "predictions.csv").relative_to(RESULTS_DIR)) if preds_src.exists() else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _future_for_best(dept: str, model: str, best_cli: str, snap) -> None:
    best = load_json(BEST_DIR / f"{slug(dept)}__{model.lower()}.json")["best_params"]
    model_class = MODELS[model][0]
    series = get_departmental_data(dept, age_group=AGE_GROUP)
    m = model_class(department=dept, age_group=AGE_GROUP, **best)
    m.fit(series)
    preds = m.predict(series, steps=4)
    step = series.index[-1] - series.index[-2]
    dates = [series.index[-1] + step * (i + 1) for i in range(4)]
    pd.DataFrame({"date": dates, "predicted": preds[:4]}).to_csv(snap / "future.csv", index=False)


def merge_manifest(new_entries: list[dict]) -> None:
    manifest = [e for e in load_manifest()
                if not e.get("label", "").endswith("(grid-search)")]
    manifest += new_entries
    save_json(RESULTS_DIR / "manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search-only", action="store_true",
                        help="Solo busca y cachea; no corre el WF completo ni toca el dashboard")
    args = parser.parse_args()

    new_entries = []
    for dept in DEPARTMENTS:
        for model in MODELS:
            best = search(dept, model)
            if not args.search_only:
                entry = full_run(dept, model, best["best_cli"])
                if entry:
                    new_entries.append(entry)

    if not args.search_only and new_entries:
        merge_manifest(new_entries)
        print(f"\n{'='*70}\nManifest actualizado con {len(new_entries)} configs grid-search.")
        print("Regenera el dashboard:  python pruebasg/build_interactive.py")
        print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
