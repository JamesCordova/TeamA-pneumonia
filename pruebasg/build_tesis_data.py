#!/usr/bin/env python
"""
Consolida en un único archivo los datos que alimentan las secciones de tesis.

El documento del Capítulo IV se genera leyendo este archivo, de modo que sus
tablas no se transcriben a mano: reejecutar los experimentos y volver a correr
este script actualiza el documento con los valores vigentes.

Salida:
    pruebasg/results/tesis/datos_capitulos.json

Uso:
    python pruebasg/build_tesis_data.py
"""

from __future__ import annotations

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

import numpy as np
import pandas as pd

from _common import RESULTS_DIR
from selector.evaluate import CANDIDATOS, REFERENCIA, comparar_todos
from selector.regiones import MUESTRA_ESTRATIFICADA, region_de
from selector.rules import COSTO_S, seleccionar

logging.disable(logging.WARNING)

GRID = RESULTS_DIR / "grid_selector"
OUT_DIR = RESULTS_DIR / "tesis"
AGE = "under5"


def _f(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), d)


def main() -> int:
    perfiles = pd.read_csv(RESULTS_DIR / "profiles" / f"perfiles_{AGE}.csv")
    perf = perfiles.set_index("departamento")
    todos = list(perf.index)
    dentro = [d for d in todos if d in MUESTRA_ESTRATIFICADA]
    fuera = [d for d in todos if d not in MUESTRA_ESTRATIFICADA]

    comps = comparar_todos(GRID, todos)

    datos = {}

    # --- 4.1 Caracterización de las series ---------------------------------
    cols = ["departamento", "region", "n_semanas", "media_casos", "cv",
            "indice_dispersion", "prop_ceros", "fuerza_tendencia",
            "fuerza_estacional", "entropia_espectral", "acf_1", "acf_52",
            "estacionaria"]
    tp = perfiles.copy()
    tp["region"] = tp["departamento"].map(region_de)
    datos["perfil"] = tp[cols].sort_values(
        ["region", "media_casos"], ascending=[True, False]).to_dict("records")

    por_region = tp.groupby("region").agg(
        n=("departamento", "count"),
        media_casos=("media_casos", "mean"),
        fuerza_estacional=("fuerza_estacional", "mean"),
        entropia=("entropia_espectral", "mean"),
        acf_52=("acf_52", "mean"),
        dispersion=("indice_dispersion", "mean"),
    ).round(3).reset_index()
    datos["perfil_por_region"] = por_region.to_dict("records")
    datos["perfil_resumen"] = {
        "n_departamentos": int(len(tp)),
        "volumen_min": _f(tp["media_casos"].min(), 1),
        "volumen_max": _f(tp["media_casos"].max(), 1),
        "dept_volumen_min": tp.loc[tp["media_casos"].idxmin(), "departamento"],
        "dept_volumen_max": tp.loc[tp["media_casos"].idxmax(), "departamento"],
        "fs_min": _f(tp["fuerza_estacional"].min()),
        "fs_max": _f(tp["fuerza_estacional"].max()),
        "dept_fs_max": tp.loc[tp["fuerza_estacional"].idxmax(), "departamento"],
        "n_no_estacionarias": int((~tp["estacionaria"]).sum()),
        "prop_ceros_max": _f(tp["prop_ceros"].max()),
        "dept_ceros_max": tp.loc[tp["prop_ceros"].idxmax(), "departamento"],
        "dispersion_min": _f(tp["indice_dispersion"].min(), 2),
        "dispersion_max": _f(tp["indice_dispersion"].max(), 2),
    }

    # --- 4.2 Desempeño comparado -------------------------------------------
    filas = []
    for dept, c in comps.items():
        fila = {"departamento": dept, "region": region_de(dept),
                "media_casos": _f(perf.loc[dept, "media_casos"], 1),
                "n_origenes": c.n_origenes}
        for m in [*CANDIDATOS, REFERENCIA]:
            fila[m] = _f(c.mae.get(m))
        fila["mejor"] = c.mejor
        fila["conjunto_empatado"] = c.conjunto_empatado
        fila["n_empatados"] = len(c.conjunto_empatado)
        filas.append(fila)
    filas.sort(key=lambda f: (f["region"], -f["media_casos"]))
    datos["desempeno"] = filas

    ganadores = pd.Series([f["mejor"] for f in filas]).value_counts().to_dict()
    datos["ganadores"] = {k: int(v) for k, v in ganadores.items()}
    datos["empates"] = {
        "n_con_3_o_mas": int(sum(1 for f in filas if f["n_empatados"] >= 3)),
        "n_con_ganador_unico": int(sum(1 for f in filas if f["n_empatados"] == 1)),
        "dept_ganador_unico": [f["departamento"] for f in filas if f["n_empatados"] == 1],
        "n_total": len(filas),
        "media_empatados": _f(np.mean([f["n_empatados"] for f in filas]), 2),
    }
    # Pertenencia al conjunto empatado por modelo
    datos["pertenencia"] = {
        m: int(sum(1 for f in filas if m in f["conjunto_empatado"]))
        for m in CANDIDATOS
    }
    # Mejora de cada familia sobre el piso de referencia
    datos["mejora_sobre_referencia"] = {
        m: _f(100 * np.mean([c.mejora_sobre_referencia.get(m, np.nan)
                             for c in comps.values()]), 1)
        for m in CANDIDATOS
    }

    # --- 4.3 Evaluación del criterio ---------------------------------------
    def _evaluar(lista):
        out = []
        for dept in lista:
            if dept not in comps:
                continue
            c = comps[dept]
            rec = seleccionar(perf.loc[dept].to_dict())
            out.append({
                "departamento": dept, "region": region_de(dept),
                "media_casos": _f(perf.loc[dept, "media_casos"], 1),
                "fuerza_estacional": _f(perf.loc[dept, "fuerza_estacional"]),
                "regla": rec.regla, "recomendado": rec.modelo,
                "mejor_real": c.mejor,
                "dentro": bool(rec.modelo in c.conjunto_empatado),
                "regret_casos": _f(c.regret(rec.modelo)),
                "regret_pct": _f(100 * c.regret_relativo(rec.modelo), 2),
            })
        return out

    def _resumen(ev, lista):
        sub = {d: comps[d] for d in lista if d in comps}
        estr = []
        for m in CANDIDATOS:
            rr = [100 * c.regret_relativo(m) for c in sub.values()]
            cob = sum(1 for c in sub.values() if m in c.conjunto_empatado)
            estr.append({"estrategia": f"siempre {m}",
                         "cobertura": f"{cob}/{len(sub)}",
                         "cobertura_pct": _f(100 * cob / len(sub), 1),
                         "regret_medio": _f(float(np.mean(rr)), 2),
                         "regret_max": _f(float(np.max(rr)), 2)})
        estr.append({"estrategia": "criterio propuesto",
                     "cobertura": f"{sum(1 for e in ev if e['dentro'])}/{len(ev)}",
                     "cobertura_pct": _f(100 * np.mean([e["dentro"] for e in ev]), 1),
                     "regret_medio": _f(float(np.mean([e["regret_pct"] for e in ev])), 2),
                     "regret_max": _f(float(np.max([e["regret_pct"] for e in ev])), 2)})
        costo_todo = sum(COSTO_S[m] for m in CANDIDATOS)
        ahorro = 1 - sum(COSTO_S[e["recomendado"]] for e in ev) / (len(ev) * costo_todo)
        return {"estrategias": estr, "ahorro_pct": _f(100 * ahorro, 1),
                "n": len(ev)}

    ev_in, ev_out = _evaluar(dentro), _evaluar(fuera)
    datos["criterio_en_muestra"] = {"filas": ev_in, **_resumen(ev_in, dentro)}
    datos["criterio_fuera_muestra"] = {"filas": ev_out, **_resumen(ev_out, fuera)}
    fallos = [e for e in ev_out if not e["dentro"]]
    datos["fallos_fuera_muestra"] = fallos

    # --- 4.4 Explicabilidad -------------------------------------------------
    fam_path = RESULTS_DIR / "shap" / "atribucion_familias.csv"
    if fam_path.exists():
        fam = pd.read_csv(fam_path)
        datos["shap_familias"] = fam.drop(
            columns=[c for c in ("top3_variables",) if c in fam.columns]
        ).to_dict("records")
    corr_path = RESULTS_DIR / "shap" / "correlacion_perfil.csv"
    if corr_path.exists():
        datos["shap_correlaciones"] = pd.read_csv(corr_path).to_dict("records")

    # --- metadatos de la corrida -------------------------------------------
    datos["configuracion"] = {
        "train_size": 520, "horizonte": 4, "paso": 4,
        "window_type": "sliding", "alpha": 0.05,
        "n_corridas": int(len(list(GRID.glob("*/metrics.json")))),
        "n_departamentos": len(todos),
        "modelos": [*CANDIDATOS, REFERENCIA],
        "costo_s": COSTO_S,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ruta = OUT_DIR / "datos_capitulos.json"
    ruta.write_text(json.dumps(datos, indent=1, ensure_ascii=False, default=str),
                    encoding="utf-8")
    print(f"Escrito: {ruta.relative_to(ROOT)}")
    print(f"  departamentos: {len(todos)}  |  corridas: {datos['configuracion']['n_corridas']}")
    print(f"  ganadores: {datos['ganadores']}")
    print(f"  empates >=3 modelos: {datos['empates']['n_con_3_o_mas']}/{datos['empates']['n_total']}")
    print(f"  criterio en muestra:    {datos['criterio_en_muestra']['estrategias'][-1]}")
    print(f"  criterio fuera muestra: {datos['criterio_fuera_muestra']['estrategias'][-1]}")
    figuras(datos)
    return 0




# ===========================================================================
# Figuras de los capítulos
# ===========================================================================

def figuras(datos: dict) -> None:
    """Genera las figuras que acompañan a las secciones de tesis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT_DIR / "figuras"
    fig_dir.mkdir(parents=True, exist_ok=True)
    COLOR_REGION = {"costa": "#4f81bd", "sierra": "#c0504d", "selva": "#9bbb59"}
    COLOR_MODELO = {"SARIMA": "#c0504d", "RandomForest": "#4f81bd",
                    "XGBoost": "#9bbb59", "LSTM": "#8064a2"}

    # --- Figura A: el espacio de perfiles, coloreado por región natural ----
    tp = pd.DataFrame(datos["perfil"])
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for reg, sub in tp.groupby("region"):
        ax.scatter(sub["media_casos"], sub["fuerza_estacional"], s=110,
                   c=COLOR_REGION.get(reg, "#777"), label=reg,
                   edgecolors="white", linewidths=1.1, zorder=3)
    for _, r in tp.iterrows():
        ax.annotate(r["departamento"].title(), (r["media_casos"], r["fuerza_estacional"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.2)
    ax.set_xscale("log")
    ax.set_xlabel("Volumen medio (casos por semana, escala logarítmica)")
    ax.set_ylabel("Fuerza estacional  $F_S$")
    ax.set_title("Espacio de perfiles de las 25 series departamentales", fontsize=11)
    ax.grid(alpha=0.25); ax.legend(title="Región natural", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "cap4_espacio_perfiles.png", dpi=140)
    plt.close(fig)

    # --- Figura B: exceso de error de cada familia, por departamento -------
    des = pd.DataFrame(datos["desempeno"])
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    datos_box, etiquetas, colores = [], [], []
    for m in ["SARIMA", "RandomForest", "XGBoost", "LSTM"]:
        mejor = des[["SARIMA", "RandomForest", "XGBoost", "LSTM"]].min(axis=1)
        datos_box.append(((des[m] - mejor) / mejor * 100).values)
        etiquetas.append(m); colores.append(COLOR_MODELO[m])
    bp = ax.boxplot(datos_box, labels=etiquetas, patch_artist=True, widths=0.55)
    for parche, col in zip(bp["boxes"], colores):
        parche.set_facecolor(col); parche.set_alpha(0.65)
    for mediana in bp["medians"]:
        mediana.set_color("black")
    ax.set_ylabel("Exceso de error sobre el mejor modelo (%)")
    ax.set_title("Distribución del exceso de error por familia, 25 departamentos", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "cap4_exceso_por_familia.png", dpi=140)
    plt.close(fig)

    # --- Figura C: tamaño del conjunto empatado ----------------------------
    conteo = des["n_empatados"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(conteo.index.astype(str), conteo.values, color="#2e5c8a", width=0.6)
    for x, v in zip(conteo.index.astype(str), conteo.values):
        ax.text(x, v + 0.2, str(v), ha="center", fontsize=9)
    ax.set_xlabel("Modelos estadísticamente indistinguibles del mejor")
    ax.set_ylabel("Número de departamentos")
    ax.set_title("Tamaño del conjunto empatado", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "cap4_conjunto_empatado.png", dpi=140)
    plt.close(fig)


    # --- Figura D: regiones de decisión del criterio ----------------------
    COLOR_MODELO_D = {"SARIMA": "#c0504d", "RandomForest": "#4f81bd",
                      "XGBoost": "#9bbb59"}
    MARCA = {"costa": "o", "sierra": "^", "selva": "s"}
    todas = pd.DataFrame(datos["criterio_en_muestra"]["filas"] +
                         datos["criterio_fuera_muestra"]["filas"])
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.axhspan(0.60, 0.80, color="#c0504d", alpha=0.07)
    ax.fill_between([40, 400], 0.05, 0.60, color="#9bbb59", alpha=0.07)
    ax.fill_between([1.5, 10], 0.05, 0.60, color="#c0504d", alpha=0.07)
    ax.axhline(0.60, color="#c0504d", ls="--", lw=1, alpha=0.7)
    ax.axvline(40, color="#9bbb59", ls="--", lw=1, alpha=0.7)
    ax.axvline(10, color="#c0504d", ls="--", lw=1, alpha=0.7)
    for _, r in todas.iterrows():
        ax.scatter(r["media_casos"], r["fuerza_estacional"], s=115,
                   c=COLOR_MODELO_D.get(r["recomendado"], "#777"),
                   marker=MARCA.get(r["region"], "o"),
                   edgecolors="white", linewidths=1.1, zorder=3)
        ax.annotate(r["departamento"].title(),
                    (r["media_casos"], r["fuerza_estacional"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log"); ax.set_xlim(1.5, 400); ax.set_ylim(0.05, 0.78)
    ax.set_xlabel("Volumen medio (casos por semana, escala logarítmica)")
    ax.set_ylabel("Fuerza estacional  $F_S$")
    ax.set_title("Regiones de decisión del criterio sobre el espacio de perfiles", fontsize=11)
    ax.grid(alpha=0.25)
    man = [plt.Line2D([], [], marker="o", ls="", color=v, label=k, markersize=9)
           for k, v in COLOR_MODELO_D.items()]
    man += [plt.Line2D([], [], marker=m, ls="", color="#555", label=k, markersize=8)
            for k, m in MARCA.items()]
    ax.legend(handles=man, loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout(); fig.savefig(fig_dir / "cap4_regiones_decision.png", dpi=140)
    plt.close(fig)

    # --- Figura E: atribución SHAP por familia ----------------------------
    if datos.get("shap_familias"):
        fam = pd.DataFrame(datos["shap_familias"])
        fam = fam[fam["modelo"] == "RandomForest"].sort_values("media_casos")
        cols = [c for c in ("calendario_estacional", "rezago_anual", "inercia_corta",
                            "memoria_media", "ventanas_moviles", "tendencia")
                if c in fam.columns]
        colores = ["#c0504d", "#e39b97", "#4f81bd", "#8fb2d8", "#9bbb59", "#cfcfcf"]
        fig, ax = plt.subplots(figsize=(8.8, 4.4))
        izq = np.zeros(len(fam))
        etiq = [f"{r.departamento.title()} ({r.media_casos:.0f})" for r in fam.itertuples()]
        for col, color in zip(cols, colores):
            ax.barh(etiq, fam[col].values, left=izq,
                    label=col.replace("_", " "), color=color)
            izq += fam[col].values
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fracción de la atribución SHAP total")
        ax.set_title("Atribución por familia de variables, bosque aleatorio "
                     "(departamentos ordenados por volumen creciente)", fontsize=11)
        ax.legend(fontsize=8, ncol=3, loc="lower right", framealpha=0.95)
        fig.tight_layout(); fig.savefig(fig_dir / "cap4_shap_familias.png", dpi=140)
        plt.close(fig)

    print(f"  figuras en {fig_dir.relative_to(ROOT)}")

if __name__ == "__main__":
    sys.exit(main())
