#!/usr/bin/env python
"""
SHAP v0: atribución de variables de los modelos de árboles sobre la muestra
estratificada, y contraste con el perfil de la serie.

La pregunta que responde: ¿el modelo se apoya en la estructura que el perfil
declara? Si en las series de estacionalidad fuerte la atribución se concentra
en el calendario y el rezago anual, y en las de volumen alto se concentra en la
inercia corta, entonces las reglas del selector describen un mecanismo y no una
coincidencia numérica.

Salidas:
  pruebasg/results/shap/atribucion_familias.csv
  pruebasg/results/shap/importancia_<DEPTO>_<modelo>.csv
  pruebasg/results/shap/figuras/<DEPTO>_<modelo>_barras.png
  pruebasg/results/shap/figuras/<DEPTO>_<modelo>_beeswarm.png

Uso:
    python pruebasg/run_shap.py
    python pruebasg/run_shap.py --modelos RandomForest --departments PUNO LIMA
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

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import RESULTS_DIR, save_json
from selector.regiones import MUESTRA_ESTRATIFICADA, region_de
from selector.shap_utils import calcular_shap, tabla_familias

OUT_DIR = RESULTS_DIR / "shap"
FIG_DIR = OUT_DIR / "figuras"


def _figura_barras(res, ruta: Path, top: int = 15) -> None:
    imp = res.importancia_global().head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color="#3b6ea5")
    ax.set_xlabel("Atribución media |SHAP| (casos)")
    ax.set_title(f"{res.departamento} — {res.modelo}", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def _figura_beeswarm(res, ruta: Path) -> None:
    import shap
    fig = plt.figure()
    shap.summary_plot(res.valores, res.X, show=False, max_display=12, plot_size=(7.2, 5.0))
    plt.title(f"{res.departamento} — {res.modelo}", fontsize=11)
    plt.tight_layout()
    plt.savefig(ruta, dpi=140)
    plt.close(fig)



def _figura_cascada(loc: dict, ruta: Path) -> None:
    """Gráfico de cascada: cómo cada variable mueve la predicción desde el valor base."""
    contribs = loc["contribuciones"]
    etiquetas = [f'{c["variable"]} = {c["valor_variable"]:g}' for c in contribs]
    valores = [c["contribucion"] for c in contribs]
    if loc["n_variables_restantes"]:
        etiquetas.append(f'otras {loc["n_variables_restantes"]} variables')
        valores.append(loc["resto_agrupado"])

    fig, ax = plt.subplots(figsize=(8.6, 0.46 * len(valores) + 1.8))
    acumulado = loc["valor_base"]
    for i, (et, v) in enumerate(zip(etiquetas, valores)):
        y = len(valores) - 1 - i
        ax.barh(y, v, left=acumulado, height=0.62,
                color="#c0504d" if v > 0 else "#4f81bd")
        ax.text(acumulado + v + (0.6 if v > 0 else -0.6), y, f"{v:+.2f}",
                va="center", ha="left" if v > 0 else "right", fontsize=8.5)
        acumulado += v
    ax.set_yticks(range(len(valores)))
    ax.set_yticklabels(etiquetas[::-1], fontsize=9)
    ax.axvline(loc["valor_base"], color="#555", ls="--", lw=1)
    ax.annotate(f'valor base {loc["valor_base"]:.1f}',
                xy=(loc["valor_base"], -0.8), fontsize=8.5, color="#555", ha="center")
    ax.axvline(loc["prediccion"], color="#222", lw=1.4)
    ax.annotate(f'predicción {loc["prediccion"]:.1f}',
                xy=(loc["prediccion"], len(valores) - 0.2), fontsize=9, ha="center")
    ax.set_xlabel("Casos semanales")
    ax.set_title(f'{loc["departamento"]} — {loc["modelo"]} — semana del {loc["semana"]}',
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--departments", nargs="+", default=None)
    parser.add_argument("--modelos", nargs="+", default=["RandomForest", "XGBoost"])
    parser.add_argument("--age_group", default="under5", choices=["under5", "60plus"])
    parser.add_argument("--sin-figuras", action="store_true")
    parser.add_argument("--local", nargs="+", default=None,
                        help="Departamentos para la explicación local de su semana pico")
    args = parser.parse_args()

    logging.disable(logging.WARNING)
    departamentos = [d.upper() for d in (args.departments or MUESTRA_ESTRATIFICADA)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    resultados = []
    for dept in departamentos:
        for modelo in args.modelos:
            print(f"  SHAP {dept:14s} {modelo}...", end="", flush=True)
            try:
                res = calcular_shap(dept, modelo=modelo, age_group=args.age_group)
            except Exception as exc:  # noqa: BLE001
                print(f" FALLO: {str(exc)[:90]}")
                continue
            resultados.append(res)
            res.importancia_global().rename("shap_medio_abs").to_csv(
                OUT_DIR / f"importancia_{dept.replace(' ', '_')}_{modelo}.csv",
                encoding="utf-8",
            )
            if not args.sin_figuras:
                slug = dept.replace(" ", "_")
                _figura_barras(res, FIG_DIR / f"{slug}_{modelo}_barras.png")
                try:
                    _figura_beeswarm(res, FIG_DIR / f"{slug}_{modelo}_beeswarm.png")
                except Exception:
                    pass
            print(" ok")

    # --- Explicación local: por qué el modelo pronosticó lo que pronosticó ---
    locales = []
    for res in resultados:
        if args.local and res.departamento not in [d.upper() for d in args.local]:
            continue
        if args.local is None and res.modelo != "RandomForest":
            continue
        try:
            i = int(res.X["lag_1"].values.argmax())
            loc = res.explicacion_local(indice=i)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! explicación local {res.departamento}: {str(exc)[:80]}")
            continue
        locales.append(loc)
        if not args.sin_figuras:
            slug = res.departamento.replace(" ", "_")
            _figura_cascada(loc, FIG_DIR / f"{slug}_{res.modelo}_local.png")
    if locales:
        save_json(OUT_DIR / "explicaciones_locales.json", locales)
        print(f"\n{len(locales)} explicaciones locales -> explicaciones_locales.json")

    if not resultados:
        print("Sin resultados.")
        return 1

    fam = tabla_familias(resultados)
    fam.insert(1, "region", fam["departamento"].map(region_de))

    perfiles = pd.read_csv(
        RESULTS_DIR / "profiles" / f"perfiles_{args.age_group}.csv"
    ).set_index("departamento")
    fam["fuerza_estacional"] = fam["departamento"].map(perfiles["fuerza_estacional"]).round(3)
    fam["media_casos"] = fam["departamento"].map(perfiles["media_casos"]).round(1)

    cols_fam = [c for c in ("calendario_estacional", "rezago_anual", "inercia_corta",
                            "memoria_media", "ventanas_moviles", "tendencia")
                if c in fam.columns]
    fam = fam[["departamento", "region", "modelo", "media_casos", "fuerza_estacional",
               *cols_fam, "top3_variables"]]
    fam = fam.sort_values(["modelo", "fuerza_estacional"], ascending=[True, False])
    fam.to_csv(OUT_DIR / "atribucion_familias.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 250)
    print("\n" + "=" * 118)
    print("ATRIBUCIÓN SHAP POR FAMILIA DE VARIABLES (fracción del total)")
    print("=" * 118)
    print(fam.drop(columns=["top3_variables"]).to_string(index=False))

    # Contraste con el perfil: ¿la atribución sigue al volumen o a la
    # estacionalidad? Se reportan ambas correlaciones porque la respuesta
    # decide cuál de las reglas del selector queda respaldada por el mecanismo.
    fam["log_volumen"] = np.log10(fam["media_casos"].clip(lower=1e-9))
    print("\nContraste con el perfil — correlación de la atribución con:")
    print(f"{'modelo':14s} {'familia':24s} {'log(volumen)':>13s} {'F_S':>9s}")
    filas_corr = []
    for modelo in fam["modelo"].unique():
        sub = fam[fam["modelo"] == modelo]
        for col in ("inercia_corta", "ventanas_moviles",
                    "calendario_estacional", "rezago_anual", "tendencia"):
            if col not in sub.columns or sub[col].std() == 0:
                continue
            r_vol = sub["log_volumen"].corr(sub[col])
            r_fs = sub["fuerza_estacional"].corr(sub[col])
            print(f"{modelo:14s} {col:24s} {r_vol:+13.3f} {r_fs:+9.3f}")
            filas_corr.append({"modelo": modelo, "familia": col,
                               "r_log_volumen": round(float(r_vol), 3),
                               "r_fuerza_estacional": round(float(r_fs), 3)})
    pd.DataFrame(filas_corr).to_csv(OUT_DIR / "correlacion_perfil.csv",
                                    index=False, encoding="utf-8")

    print(f"\nGuardado en {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
