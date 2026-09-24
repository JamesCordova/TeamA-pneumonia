#!/usr/bin/env python
r"""
Runner de experimentos walk-forward
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime

# Salida robusta en Windows
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from _common import (
    REPORTS_DIR,
    RESULTS_DIR,
    ROOT,
    load_json,
    save_json,
    slug,
)

AGE_GROUP = "under5"
COMMON_ARGS = ["--horizon", "4", "--step", "4"]
DEPARTMENTS = ["AMAZONAS", "LIMA"]

# Modelos + variación de parámetros. refit_every alto en SARIMA/RNN/Prophet
# para que el walk-forward sea tratable en tiempo (si no, reentrena cada paso).
MODELS = [
    {"label": "SeasonalNaive (baseline)", "model": "SeasonalNaive", "args": []},
    {"label": "RandomForest (default)", "model": "RandomForest", "args": []},
    {"label": "RandomForest (lags+windows)", "model": "RandomForest",
     "args": ["--lags", "1", "2", "4", "8", "26", "52",
              "--windows", "4", "13", "26", "52"]},
    {"label": "XGBoost (tuned)", "model": "XGBoost",
     "args": ["--n_estimators", "500", "--learning_rate", "0.01",
              "--max_depth", "3", "--colsample_bytree", "0.7"]},
    {"label": "SARIMA (Fourier K=10)", "model": "SARIMA",
     "args": ["--n_fourier_terms", "10", "--refit_every", "13"]},
    {"label": "LSTM", "model": "LSTM",
     "args": ["--epochs", "40", "--refit_every", "52"]},
    {"label": "GRU", "model": "GRU",
     "args": ["--epochs", "40", "--refit_every", "52"]},
    # Prophet requiere CmdStan/RTools compilado (ver README). Re-habilitar cuando
    # el backend de Stan esté configurado:
    # {"label": "Prophet", "model": "Prophet", "args": ["--refit_every", "13"]},
]

EXPERIMENTS = [
    {**m, "department": dept} for dept in DEPARTMENTS for m in MODELS
]


def _reports_paths(department: str, model: str):
    """Rutas donde run_walkforward.py deja los archivos de este dept+modelo."""
    out_dir = REPORTS_DIR / department.upper() / AGE_GROUP
    metrics = out_dir / f"{model.lower()}_walkforward_metrics.json"
    preds = out_dir / f"{model}_predictions.csv"
    return metrics, preds


def build_command(exp: dict) -> list[str]:
    """Lista de argumentos para subprocess (usa el mismo intérprete/venv)."""
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
    """Versión legible del comando (oculta la ruta absoluta del python)."""
    parts = ["python", "scripts/run_walkforward.py", *cmd[2:]]
    return " ".join(parts)


def run_one(exp: dict, dry_run: bool) -> dict | None:
    cmd = build_command(exp)
    shown = command_display(cmd)
    print("\n" + "=" * 78)
    print(f"[{exp['label']}]  {exp['department']} / {exp['model']}")
    print(f"  $ {shown}")
    print("=" * 78)

    if dry_run:
        return None

    child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=child_env,
    )
    if result.returncode != 0:
        print("  [FALLO] (returncode "
              f"{result.returncode}). Ultimas lineas de error:")
        print("\n".join(result.stdout.splitlines()[-8:]))
        print("\n".join(result.stderr.splitlines()[-8:]))
        return {
            "label": exp["label"], "department": exp["department"].upper(),
            "model": exp["model"], "age_group": AGE_GROUP,
            "command": shown, "returncode": result.returncode,
            "ok": False, "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    # Snapshot de los archivos generados
    metrics_src, preds_src = _reports_paths(exp["department"], exp["model"])
    snap = slug(f"{exp['department']}-{exp['label']}")
    dest_dir = RESULTS_DIR / snap
    dest_dir.mkdir(parents=True, exist_ok=True)

    metrics_dest = dest_dir / "metrics.json"
    preds_dest = dest_dir / "predictions.csv"
    config = {}
    if metrics_src.exists():
        shutil.copy2(metrics_src, metrics_dest)
        config = load_json(metrics_dest).get("config", {})
    else:
        print(f"  ! No se encontró {metrics_src} (¿cambió el nombre del modelo?)")
    if preds_src.exists():
        shutil.copy2(preds_src, preds_dest)

    print(f"  [OK] snapshot en pruebasg/results/{snap}/")
    return {
        "label": exp["label"],
        "department": exp["department"].upper(),
        "model": exp["model"],
        "age_group": AGE_GROUP,
        "command": shown,
        "returncode": 0,
        "ok": True,
        "config": config,
        "metrics_file": str(metrics_dest.relative_to(RESULTS_DIR)) if metrics_src.exists() else None,
        "pred_file": str(preds_dest.relative_to(RESULTS_DIR)) if preds_src.exists() else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra los comandos sin ejecutarlos")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for exp in EXPERIMENTS:
        entry = run_one(exp, args.dry_run)
        if entry is not None:
            manifest.append(entry)

    if not args.dry_run:
        save_json(RESULTS_DIR / "manifest.json", manifest)
        ok = sum(1 for e in manifest if e.get("ok"))
        print(f"\n{'=' * 78}")
        print(f"Terminado: {ok}/{len(manifest)} experimentos OK.")
        print("Ahora genera el dashboard:")
        print("    python pruebasg/build_dashboard.py")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
