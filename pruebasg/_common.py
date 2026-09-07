
from __future__ import annotations

import json
import re
from pathlib import Path

# Raíz del repo = carpeta padre de pruebasg/
PRUEBAS_DIR = Path(__file__).resolve().parent
ROOT = PRUEBAS_DIR.parent
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = PRUEBAS_DIR / "results"
MANIFEST_PATH = RESULTS_DIR / "manifest.json"

# Métricas que mostramos en la tabla (clave interna -> encabezado)
METRIC_HEADERS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "smape": "SMAPE (%)",
    "mda": "MDA (%)",
}


def slug(text: str) -> str:
    """Convierte una etiqueta en un nombre de carpeta seguro."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "exp"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    data = load_json(MANIFEST_PATH)
    return data if isinstance(data, list) else data.get("experiments", [])
