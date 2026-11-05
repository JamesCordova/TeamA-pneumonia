#!/usr/bin/env python
r"""
Chequeo: la comparación de modelos puntúa 2020-2021, que la configuración
declara excluido.

Qué pasa
--------
`scripts/run_walkforward.py` corre con `--exclude_covid` activado por omisión
—«Exclude 2020-2021 from reported metrics (default: True)»—. El entrenamiento
y el pronóstico sí atraviesan esas fechas; lo que la opción cambia es qué
semanas cuentan para la puntuación. Por eso `metrics.json` reporta el MAE sin
esas semanas, mientras que `predictions.csv` guarda todas las predicciones,
incluidas las de 2020-2021.

`pruebasg/selector/evaluate.py` lee `predictions.csv` y promedia todo. De ahí
salen el conjunto empatado, la evaluación del criterio y la tabla comparativa
del capítulo IV. Es decir: el documento puntúa 26 orígenes que la propia
configuración del experimento declara fuera.

Para Lima / XGBoost, horizonte 1:

    MAE sobre los 183 orígenes .................. 19,829   (lo que usa la tesis)
    MAE sobre los 157 sin 2020-2021 ............. 20,571   (lo que dice metrics.json)

El sentido del desvío es el esperado: durante la pandemia la notificación cayó,
los conteos fueron pequeños y los errores absolutos también, de modo que
incluir esas semanas baja el MAE de todos los modelos.

Para qué sirve este script
--------------------------
Cuantifica el efecto sobre las cifras publicadas antes de decidir si se
corrigen. No modifica nada: solo imprime la comparación lado a lado y, con
`--csv`, la guarda.

Uso:
    python pruebasg/chequeo_exclusion_covid.py
    python pruebasg/chequeo_exclusion_covid.py --csv pruebasg/results/selector/chequeo_covid.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from selector.evaluate import cargar_errores, comparar
from selector.rules import seleccionar

logging.disable(logging.WARNING)

GRID = RESULTS_DIR / "grid_selector"
INICIO, FIN = "2020-01-01", "2021-12-31"


def sin_pandemia(errores: pd.DataFrame) -> pd.DataFrame:
    fuera = (errores.index >= INICIO) & (errores.index <= FIN)
    return errores[~fuera]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--age_group", default="under5")
    p.add_argument("--horizonte", type=int, default=1)
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    perfiles = pd.read_csv(
        RESULTS_DIR / "profiles" / f"perfiles_{args.age_group}.csv"
    ).set_index("departamento")

    filas = []
    for dept in sorted(perfiles.index):
        try:
            err = cargar_errores(GRID, dept, horizonte=args.horizonte)
        except FileNotFoundError:
            continue
        con = comparar(err, dept, horizonte=args.horizonte)
        sin = comparar(sin_pandemia(err), dept, horizonte=args.horizonte)
        rec = seleccionar(perfiles.loc[dept].to_dict())
        filas.append({
            "departamento": dept,
            "origenes_con": con.n_origenes, "origenes_sin": sin.n_origenes,
            "recomendado": rec.modelo, "regla": rec.regla,
            "mejor_con": con.mejor, "mejor_sin": sin.mejor,
            "empatados_con": len(con.conjunto_empatado),
            "empatados_sin": len(sin.conjunto_empatado),
            "conjunto_con": " | ".join(con.conjunto_empatado),
            "conjunto_sin": " | ".join(sin.conjunto_empatado),
            "dentro_con": rec.modelo in con.conjunto_empatado,
            "dentro_sin": rec.modelo in sin.conjunto_empatado,
            "mae_rec_con": round(con.mae[rec.modelo], 3),
            "mae_rec_sin": round(sin.mae[rec.modelo], 3),
            "regret_con_pct": round(100 * con.regret_relativo(rec.modelo), 2),
            "regret_sin_pct": round(100 * sin.regret_relativo(rec.modelo), 2),
            "lstm_empata_con": "LSTM" in con.conjunto_empatado,
            "lstm_empata_sin": "LSTM" in sin.conjunto_empatado,
            "sarima_empata_con": "SARIMA" in con.conjunto_empatado,
            "sarima_empata_sin": "SARIMA" in sin.conjunto_empatado,
        })

    if not filas:
        print("Sin resultados en", GRID)
        return 1
    t = pd.DataFrame(filas)
    n = len(t)

    def linea(etiqueta, con, sin):
        print(f"  {etiqueta:38s} {con:>14}   {sin:>14}")

    print("=" * 74)
    print(f"EFECTO DE EXCLUIR {INICIO[:4]}-{FIN[:4]} DE LA PUNTUACIÓN "
          f"(h={args.horizonte}, {n} departamentos)")
    print("=" * 74)
    print(f"  {'':38s} {'como se publica':>14}   {'excluyendo':>14}")
    linea("orígenes puntuados por serie",
          int(t["origenes_con"].iloc[0]), int(t["origenes_sin"].iloc[0]))
    linea("series con 3 o más familias empatadas",
          f"{int((t.empatados_con >= 3).sum())}/{n}", f"{int((t.empatados_sin >= 3).sum())}/{n}")
    linea("series con ganador único",
          f"{int((t.empatados_con == 1).sum())}/{n}", f"{int((t.empatados_sin == 1).sum())}/{n}")
    linea("media de familias empatadas",
          f"{t.empatados_con.mean():.2f}", f"{t.empatados_sin.mean():.2f}")
    linea("LSTM dentro del conjunto empatado",
          f"{int(t.lstm_empata_con.sum())}/{n}", f"{int(t.lstm_empata_sin.sum())}/{n}")
    linea("SARIMA dentro del conjunto empatado",
          f"{int(t.sarima_empata_con.sum())}/{n}", f"{int(t.sarima_empata_sin.sum())}/{n}")
    linea("cobertura del criterio",
          f"{int(t.dentro_con.sum())}/{n}", f"{int(t.dentro_sin.sum())}/{n}")
    linea("exceso de error medio",
          f"{t.regret_con_pct.mean():.2f} %", f"{t.regret_sin_pct.mean():.2f} %")
    linea("cambia el mejor observado en",
          "—", f"{int((t.mejor_con != t.mejor_sin).sum())} series")
    linea("cambia el conjunto empatado en",
          "—", f"{int((t.conjunto_con != t.conjunto_sin).sum())} series")

    cam = t[(t.conjunto_con != t.conjunto_sin) | (t.dentro_con != t.dentro_sin)]
    if len(cam):
        print("\nSeries en las que cambia el conjunto empatado:")
        pd.set_option("display.width", 220)
        print(cam[["departamento", "mejor_con", "mejor_sin",
                   "conjunto_con", "conjunto_sin", "dentro_con", "dentro_sin"]]
              .to_string(index=False))

    print("\nLectura: la conclusión central —el criterio cae dentro del conjunto")
    print("empatado y los conjuntos son amplios— se mantiene. Lo que cambia son")
    print("dos cifras citadas en el documento: cuántas series tienen tres o más")
    print("familias empatadas y en cuántas queda LSTM fuera del empate.")

    if args.csv:
        destino = Path(args.csv)
        destino.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(destino, index=False, encoding="utf-8")
        print(f"\nDetalle por serie -> {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
