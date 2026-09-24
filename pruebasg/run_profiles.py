#!/usr/bin/env python
"""
Perfila las series semanales de todos los departamentos.

Genera el insumo del selector automático de modelos:
    pruebasg/results/profiles/perfiles_<age_group>.csv   (tabla completa)
    pruebasg/results/profiles/perfiles_<age_group>.json  (mismo contenido)

Uso:
    python pruebasg/run_profiles.py
    python pruebasg/run_profiles.py --age_group 60plus
    python pruebasg/run_profiles.py --solo-muestra     # solo los estratificados
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PRUEBAS_DIR = Path(__file__).resolve().parent
ROOT = PRUEBAS_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PRUEBAS_DIR))

import pandas as pd

from _common import RESULTS_DIR, save_json
from selector.profiler import perfilar_departamentos
from selector.regiones import MUESTRA_ESTRATIFICADA, region_de

PROFILES_DIR = RESULTS_DIR / "profiles"

# Columnas que se muestran en el resumen por consola (las demás quedan en el CSV).
COLS_RESUMEN = [
    "departamento", "region", "n_semanas", "media_casos", "cv",
    "indice_dispersion", "fuerza_estacional", "fuerza_tendencia",
    "entropia_espectral", "acf_1", "acf_52", "estacionaria",
    "cat_volumen", "cat_estacionalidad", "cat_entropia",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--age_group", default="under5", choices=["under5", "60plus"])
    parser.add_argument("--solo-muestra", action="store_true",
                        help="Perfilar solo los departamentos de la muestra estratificada")
    args = parser.parse_args()

    from pneumonia.models.utils import get_available_departments

    departamentos = (
        MUESTRA_ESTRATIFICADA if args.solo_muestra else get_available_departments()
    )
    print(f"Perfilando {len(departamentos)} departamentos ({args.age_group})...\n")

    df = perfilar_departamentos(departamentos, age_group=args.age_group)
    df.insert(1, "region", df["departamento"].map(region_de))
    df.insert(2, "en_muestra", df["departamento"].isin(MUESTRA_ESTRATIFICADA))
    df = df.sort_values(["region", "media_casos"], ascending=[True, False])

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROFILES_DIR / f"perfiles_{args.age_group}.csv"
    json_path = PROFILES_DIR / f"perfiles_{args.age_group}.json"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    save_json(json_path, df.to_dict(orient="records"))

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[COLS_RESUMEN].to_string(index=False))

    print(f"\nGuardado: {csv_path.relative_to(ROOT)}")
    print(f"Guardado: {json_path.relative_to(ROOT)}")

    print("\nCobertura de categorías del perfilador (muestra estratificada):")
    muestra = df[df["en_muestra"]]
    for col in ("cat_volumen", "cat_estacionalidad", "cat_entropia",
                "cat_dispersion", "cat_estacionariedad"):
        conteo = muestra[col].value_counts().to_dict()
        print(f"  {col:22s} {conteo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
