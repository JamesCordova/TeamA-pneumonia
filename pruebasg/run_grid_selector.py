#!/usr/bin/env python
r"""
Grilla walk-forward sobre la muestra estratificada de departamentos.

Es la evidencia empírica que el selector automático necesita: para cada perfil
de serie, cuál familia de modelo gana realmente. A diferencia de run_grid.py
(2 departamentos, exploración de configuraciones), aquí se fija UNA
configuración por familia y se barre la muestra estratificada completa, para
que la comparación entre departamentos sea limpia.

Es reanudable: un experimento cuyo snapshot ya existe se omite, salvo --force.

Uso:
    python pruebasg/run_grid_selector.py
    python pruebasg/run_grid_selector.py --dry-run
    python pruebasg/run_grid_selector.py --departments LORETO PUNO
    python pruebasg/run_grid_selector.py --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PRUEBAS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PRUEBAS_DIR))

from _common import REPORTS_DIR, RESULTS_DIR, ROOT, load_json, save_json, slug
from selector.regiones import MUESTRA_ESTRATIFICADA

AGE_GROUP = "under5"
COMMON_ARGS = ["--horizon", "4", "--step", "4", "--train_size", "520"]

# Snapshots y manifiesto propios, separados de los de run_grid.py, para no
# mezclar la exploración de configuraciones con la corrida comparativa.
GRID_DIR = RESULTS_DIR / "grid_selector"
MANIFEST = GRID_DIR / "manifest.json"

# Una configuración por familia de modelo. Las de SARIMA/RF/XGBoost son las que
# ganaron en la exploración previa (ver pruebasg/results/manifest.json); el
# grid search no las superó, así que se toman como representantes de su familia.
#
# SeasonalNaive se corre como piso de referencia, NO como candidato del
# selector: sirve para saber cuánto aporta cada familia sobre la persistencia
# estacional, pero recomendarlo nunca es una decisión de ingeniería útil.
MODELOS = [
    {"label": "SeasonalNaive", "model": "SeasonalNaive", "args": [], "candidato": False},
    {"label": "SARIMA-Fourier", "model": "SARIMA",
     "args": ["--n_fourier_terms", "10", "--refit_every", "13"], "candidato": True},
    {"label": "RandomForest", "model": "RandomForest",
     "args": ["--lags", "1", "2", "4", "8", "26", "52",
              "--windows", "4", "13", "26", "52"], "candidato": True},
    {"label": "XGBoost", "model": "XGBoost",
     "args": ["--n_estimators", "500", "--learning_rate", "0.01",
              "--max_depth", "3", "--colsample_bytree", "0.7"], "candidato": True},
    {"label": "LSTM", "model": "LSTM",
     "args": ["--epochs", "40", "--refit_every", "52"], "candidato": True},
]


def _reports_paths(department: str, model: str):
    out_dir = REPORTS_DIR / department.upper() / AGE_GROUP
    return (out_dir / f"{model.lower()}_walkforward_metrics.json",
            out_dir / f"{model}_predictions.csv")


def build_command(exp: dict) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_walkforward.py"),
        "--department", exp["department"],
        "--model", exp["model"],
        "--age_group", AGE_GROUP,
        *COMMON_ARGS,
        *exp["args"],
    ]


def command_display(cmd: list[str]) -> str:
    return " ".join(["python", "scripts/run_walkforward.py", *cmd[2:]])


def run_one(exp: dict, dry_run: bool, force: bool) -> dict | None:
    snap = slug(f"{exp['department']}-{exp['label']}")
    dest_dir = GRID_DIR / snap
    metrics_dest = dest_dir / "metrics.json"

    cmd = build_command(exp)
    shown = command_display(cmd)
    print("\n" + "=" * 78)
    print(f"[{exp['label']}]  {exp['department']}")
    print(f"  $ {shown}")

    if metrics_dest.exists() and not force:
        print("  [OMITIDO] snapshot ya existe (usa --force para rehacerlo)")
        entry = load_json(metrics_dest)
        return {
            "label": exp["label"], "department": exp["department"].upper(),
            "model": exp["model"], "age_group": AGE_GROUP, "candidato": exp["candidato"],
            "command": shown, "ok": True, "reused": True,
            "config": entry.get("config", {}),
            "metrics_file": f"{snap}/metrics.json",
            "pred_file": f"{snap}/predictions.csv",
        }

    if dry_run:
        return None

    t0 = time.time()
    child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=child_env)
    dur = time.time() - t0

    if result.returncode != 0:
        print(f"  [FALLO] returncode={result.returncode} ({dur:.0f}s). Últimas líneas:")
        print("\n".join(result.stdout.splitlines()[-6:]))
        print("\n".join(result.stderr.splitlines()[-6:]))
        return {
            "label": exp["label"], "department": exp["department"].upper(),
            "model": exp["model"], "age_group": AGE_GROUP, "candidato": exp["candidato"],
            "command": shown, "returncode": result.returncode, "ok": False,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    metrics_src, preds_src = _reports_paths(exp["department"], exp["model"])
    dest_dir.mkdir(parents=True, exist_ok=True)
    config = {}
    if metrics_src.exists():
        shutil.copy2(metrics_src, metrics_dest)
        config = load_json(metrics_dest).get("config", {})
    else:
        print(f"  ! No se encontró {metrics_src}")
    if preds_src.exists():
        shutil.copy2(preds_src, dest_dir / "predictions.csv")

    print(f"  [OK] {dur:.0f}s -> pruebasg/results/grid_selector/{snap}/")
    return {
        "label": exp["label"], "department": exp["department"].upper(),
        "model": exp["model"], "age_group": AGE_GROUP, "candidato": exp["candidato"],
        "command": shown, "returncode": 0, "ok": True, "reused": False,
        "config": config, "duracion_s": round(dur, 1),
        "metrics_file": f"{snap}/metrics.json",
        "pred_file": f"{snap}/predictions.csv",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Rehacer experimentos con snapshot existente")
    parser.add_argument("--departments", nargs="+", default=None,
                        help="Subconjunto de departamentos (por defecto: la muestra estratificada)")
    args = parser.parse_args()

    departamentos = [d.upper() for d in (args.departments or MUESTRA_ESTRATIFICADA)]
    experimentos = [{**m, "department": d} for d in departamentos for m in MODELOS]

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Grilla: {len(departamentos)} departamentos x {len(MODELOS)} modelos "
          f"= {len(experimentos)} corridas")

    t0 = time.time()
    manifest = []
    for exp in experimentos:
        entry = run_one(exp, args.dry_run, args.force)
        if entry is not None:
            manifest.append(entry)
            if not args.dry_run:
                save_json(MANIFEST, manifest)   # guardado incremental

    if not args.dry_run:
        ok = sum(1 for e in manifest if e.get("ok"))
        print(f"\n{'=' * 78}")
        print(f"Terminado: {ok}/{len(manifest)} corridas OK en {(time.time()-t0)/60:.1f} min")
        print(f"Manifiesto: {MANIFEST.relative_to(ROOT)}")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
