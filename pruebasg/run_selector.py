#!/usr/bin/env python
"""
Aplica el selector v1 a la muestra estratificada y mide su acierto.

Métricas del selector (no del modelo):
  - Cobertura: fracción de series en las que la recomendación cae dentro del
    conjunto estadísticamente empatado con el mejor modelo. Es la métrica
    principal: dado que los tres modelos clásicos son indistinguibles en la
    mayoría de series, acertar el mínimo exacto no es un objetivo alcanzable
    ni informativo.
  - Regret: exceso de MAE de la recomendación respecto del mejor modelo.
  - Ahorro: costo de entrenar solo el modelo recomendado frente a evaluar
    todas las familias.

Salidas:
  pruebasg/results/selector/evaluacion_selector.csv
  pruebasg/results/selector/comparacion_estrategias.csv
  pruebasg/results/selector/reglas.csv

Uso:
    python pruebasg/run_selector.py
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

import numpy as np
import pandas as pd

from _common import RESULTS_DIR
from selector.evaluate import CANDIDATOS, comparar_todos
from selector.regiones import MUESTRA_ESTRATIFICADA, region_de
from selector.rules import DESCRIPCION_REGLAS, seleccionar

GRID_DIR = RESULTS_DIR / "grid_selector"
OUT_DIR = RESULTS_DIR / "selector"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--age_group", default="under5", choices=["under5", "60plus"])
    parser.add_argument("--horizonte", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    perfiles = pd.read_csv(
        RESULTS_DIR / "profiles" / f"perfiles_{args.age_group}.csv"
    ).set_index("departamento")

    comps = comparar_todos(GRID_DIR, MUESTRA_ESTRATIFICADA,
                           horizonte=args.horizonte, alpha=args.alpha)
    if not comps:
        print("No hay resultados en", GRID_DIR)
        return 1

    filas = []
    for dept, comp in comps.items():
        perfil = perfiles.loc[dept].to_dict()
        rec = seleccionar(perfil)
        dentro = rec.modelo in comp.conjunto_empatado
        filas.append({
            "departamento": dept,
            "region": region_de(dept),
            "media_casos": round(perfil["media_casos"], 1),
            "fuerza_estacional": round(perfil["fuerza_estacional"], 3),
            "regla": rec.regla,
            "recomendado": rec.modelo,
            "mejor_real": comp.mejor,
            "conjunto_empatado": " | ".join(comp.conjunto_empatado),
            "dentro_del_empate": dentro,
            "mae_recomendado": round(comp.mae[rec.modelo], 3),
            "mae_mejor": round(comp.mae[comp.mejor], 3),
            "regret": round(comp.regret(rec.modelo), 3),
            "regret_rel_pct": round(comp.regret_relativo(rec.modelo) * 100, 2),
            "descarta": " | ".join(rec.descartados),
            "costo_s": rec.costo_estimado_s,
            "costo_todo_s": rec.costo_evaluar_todo_s,
        })

    ev = pd.DataFrame(filas).sort_values(["region", "media_casos"],
                                         ascending=[True, False])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT_DIR / "evaluacion_selector.csv", index=False, encoding="utf-8")

    # --- Comparación contra estrategias fijas y contra el oráculo -----------
    estrategias = []
    for modelo in CANDIDATOS:
        regrets = [c.regret_relativo(modelo) * 100 for c in comps.values()]
        dentro = sum(1 for c in comps.values() if modelo in c.conjunto_empatado)
        estrategias.append({
            "estrategia": f"siempre {modelo}",
            "cobertura_pct": round(100 * dentro / len(comps), 1),
            "regret_medio_pct": round(float(np.mean(regrets)), 2),
            "regret_max_pct": round(float(np.max(regrets)), 2),
            "costo_total_s": round(len(comps) * {"SARIMA": 24.0, "RandomForest": 38.3,
                                                 "XGBoost": 36.0, "LSTM": 81.6}[modelo], 0),
        })
    estrategias.append({
        "estrategia": "selector v1",
        "cobertura_pct": round(100 * ev["dentro_del_empate"].mean(), 1),
        "regret_medio_pct": round(float(ev["regret_rel_pct"].mean()), 2),
        "regret_max_pct": round(float(ev["regret_rel_pct"].max()), 2),
        "costo_total_s": round(float(ev["costo_s"].sum()), 0),
    })
    estrategias.append({
        "estrategia": "evaluar todo y elegir (oráculo)",
        "cobertura_pct": 100.0,
        "regret_medio_pct": 0.0,
        "regret_max_pct": 0.0,
        "costo_total_s": round(float(ev["costo_todo_s"].sum()), 0),
    })
    comp_df = pd.DataFrame(estrategias)
    comp_df.to_csv(OUT_DIR / "comparacion_estrategias.csv", index=False, encoding="utf-8")

    reglas_df = pd.DataFrame(DESCRIPCION_REGLAS,
                             columns=["id", "nombre", "condicion", "recomienda", "fundamento"])
    reglas_df.to_csv(OUT_DIR / "reglas.csv", index=False, encoding="utf-8")

    # --- Reporte por consola ------------------------------------------------
    pd.set_option("display.width", 250)
    print("=" * 110)
    print(f"SELECTOR v1 — muestra estratificada, horizonte h={args.horizonte}, alpha={args.alpha}")
    print("=" * 110)
    print(ev[["departamento", "region", "media_casos", "fuerza_estacional", "regla",
              "recomendado", "mejor_real", "dentro_del_empate", "regret_rel_pct"]]
          .to_string(index=False))

    print("\n" + "=" * 110)
    print("COMPARACIÓN DE ESTRATEGIAS")
    print("=" * 110)
    print(comp_df.to_string(index=False))

    n = len(ev)
    print(f"\nCobertura del selector: {int(ev['dentro_del_empate'].sum())}/{n} series "
          f"dentro del conjunto empatado ({100 * ev['dentro_del_empate'].mean():.1f} %)")
    print(f"Regret medio: {ev['regret_rel_pct'].mean():.2f} %   "
          f"máximo: {ev['regret_rel_pct'].max():.2f} % ({ev.loc[ev['regret_rel_pct'].idxmax(), 'departamento']})")
    ahorro = 1 - ev["costo_s"].sum() / ev["costo_todo_s"].sum()
    print(f"Ahorro de cómputo frente a evaluar las cuatro familias: {100 * ahorro:.1f} %")
    print(f"\nGuardado en {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
