#!/usr/bin/env python
"""
Figuras de la Etapa 6 del Capítulo III (visualización comparativa de resultados).

Responde a la observación de que el apartado 3.9.2 describe representaciones
gráficas sin mostrarlas. Genera las tres que el texto menciona —el gráfico de
serie con predicciones, la comparación de modelos y el mapa de calor del error—
a partir de los resultados reales de la validación.

Salidas:
    pruebasg/results/tesis/figuras/cap3_*.png

Uso:
    python pruebasg/build_figuras_cap3.py
    python pruebasg/build_figuras_cap3.py --departamento LORETO
"""

from __future__ import annotations

import argparse
import json
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

from _common import RESULTS_DIR

logging.disable(logging.WARNING)

GRID = RESULTS_DIR / "grid_selector"
FIG_DIR = RESULTS_DIR / "tesis" / "figuras"
COLOR = {"SARIMA": "#c0504d", "RandomForest": "#4f81bd", "XGBoost": "#9bbb59",
         "LSTM": "#8064a2", "SeasonalNaive": "#999999"}
ORDEN = ["SeasonalNaive", "SARIMA", "RandomForest", "XGBoost", "LSTM"]


def cargar(dept: str) -> dict[str, pd.DataFrame]:
    """Predicciones de cada modelo para un departamento."""
    out = {}
    for sub in sorted(GRID.iterdir()):
        met, pred = sub / "metrics.json", sub / "predictions.csv"
        if not (met.exists() and pred.exists()):
            continue
        info = json.loads(met.read_text(encoding="utf-8-sig"))
        if info.get("department", "").upper() != dept.upper():
            continue
        out[info["model"]] = pd.read_csv(pred, parse_dates=["date"])
    return out


def fig_backtest(dept: str, datos: dict, desde="2021-01-01") -> None:
    """Serie observada frente a las predicciones de la validación."""
    from pneumonia.models.utils import get_departmental_data
    serie = get_departmental_data(dept)
    serie = serie[serie.index >= desde]

    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    ax.plot(serie.index, serie.values, color="#222", lw=1.4,
            label="Observado", zorder=4)
    for modelo in ("SARIMA", "RandomForest", "XGBoost"):
        if modelo not in datos:
            continue
        d = datos[modelo]
        d = d[(d["horizon"] == 1) & (d["date"] >= desde)]
        ax.plot(d["date"], d["predicted"], lw=1.1, ls="--",
                color=COLOR[modelo], label=f"{modelo} (h=1)", alpha=0.9)
    ax.set_ylabel("Casos semanales")
    ax.set_title(f"Validación de origen móvil — {dept.title()}, menores de 5 años",
                 fontsize=11)
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cap3_backtest.png", dpi=140)
    plt.close(fig)


def fig_comparacion(dept: str, datos: dict) -> None:
    """Comparación del error de todos los modelos, un horizonte a la vez."""
    filas = []
    for modelo, d in datos.items():
        for h in (1, 4):
            sub = d[d["horizon"] == h]
            if sub.empty:
                continue
            e = (sub["actual"] - sub["predicted"]).abs()
            filas.append({"modelo": modelo, "h": h, "mae": float(e.mean())})
    df = pd.DataFrame(filas)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, h in zip(axes, (1, 4)):
        sub = df[df["h"] == h].set_index("modelo").reindex(
            [m for m in ORDEN if m in df["modelo"].values])
        colores = [COLOR[m] for m in sub.index]
        ax.barh(sub.index, sub["mae"], color=colores, height=0.6)
        for i, v in enumerate(sub["mae"]):
            ax.text(v + sub["mae"].max() * 0.02, i, f"{v:.2f}",
                    va="center", fontsize=8.5)
        ax.set_xlabel("MAE (casos)")
        ax.set_title(f"Horizonte h = {h} semana{'s' if h > 1 else ''}", fontsize=10.5)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlim(0, sub["mae"].max() * 1.18)
    axes[0].invert_yaxis()
    fig.suptitle(f"Comparación de modelos — {dept.title()}", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cap3_comparacion.png", dpi=140)
    plt.close(fig)


def fig_heatmap(dept: str, datos: dict) -> None:
    """Mapa de calor del error por año y modelo."""
    filas = []
    for modelo, d in datos.items():
        sub = d[d["horizon"] == 1].copy()
        sub["anio"] = sub["date"].dt.year
        sub["err"] = (sub["actual"] - sub["predicted"]).abs()
        for anio, g in sub.groupby("anio"):
            filas.append({"modelo": modelo, "anio": int(anio), "mae": float(g["err"].mean())})
    df = pd.DataFrame(filas)
    piv = df.pivot(index="modelo", columns="anio", values="mae")
    piv = piv.reindex([m for m in ORDEN if m in piv.index])

    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=8, rotation=90)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > np.nanmax(piv.values) * 0.6 else "#222")
    fig.colorbar(im, ax=ax, label="MAE (casos)", fraction=0.025, pad=0.01)
    ax.set_title(f"Evolución del error por año y modelo — {dept.title()} (h = 1)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cap3_heatmap.png", dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--departamento", default="AMAZONAS")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    datos = cargar(args.departamento)
    if not datos:
        print(f"Sin resultados para {args.departamento}")
        return 1
    print(f"{args.departamento}: {len(datos)} modelos")
    fig_backtest(args.departamento, datos)
    fig_comparacion(args.departamento, datos)
    fig_heatmap(args.departamento, datos)
    print(f"Figuras en {FIG_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
