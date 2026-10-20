#!/usr/bin/env python
r"""
Frontend v2.0 — Pronóstico de neumonía por departamento (Perú).

Sucede a frontend_neumonia.html. Conserva el mapa coroplético y añade lo que
distingue a esta investigación de una plataforma de visualización corriente:

  Panel 1 — Mapa interactivo por departamento (tasa por 100 000 habitantes).
  Panel 2 — Criterio de selección automática de modelo: el perfil de cada serie,
            la regla que se activa, la familia recomendada y si pertenece al
            conjunto de modelos estadísticamente equivalentes al mejor.
  Panel 3 — Modelado por departamento: observado frente a predicho con banda de
            incertidumbre empírica al 80 %, fecha de emisión y horizonte.
  Panel 4 — Explicabilidad SHAP: atribución global por familia de variables y
            descomposición local de una predicción concreta.
  Panel 5 — Comparación de modelos y evaluación del criterio.

Los paneles 2 y 4 son los que no existen en plataformas equivalentes.

Uso:
    python pruebasg/build_frontend_v2.py
    python pruebasg/build_frontend_v2.py --anios 3
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import date
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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.offline import get_plotlyjs

from _common import RESULTS_DIR

logging.disable(logging.WARNING)

MAROON = "#841617"
GRID = RESULTS_DIR / "grid_selector"
SHAP_DIR = RESULTS_DIR / "shap"
SEL_DIR = RESULTS_DIR / "selector"
PROF_DIR = RESULTS_DIR / "profiles"
TESIS_DIR = RESULTS_DIR / "tesis"
RATES_CSV = ROOT / "reports" / "tables" / "departmental_annual_variation_rates.csv"
GEOJSON = PRUEBAS_DIR / "peru_departamentos.geojson"

CFG = {"displaylogo": False, "responsive": True}
CFG_MAP = {"displaylogo": False, "responsive": True, "scrollZoom": False}

COLOR = {"SARIMA": "#c0504d", "RandomForest": "#4f81bd", "XGBoost": "#6f9c3d",
         "LSTM": "#8064a2", "SeasonalNaive": "#9a9a9a"}
CANDIDATOS = ["SARIMA", "RandomForest", "XGBoost", "LSTM"]
MEASURES = [
    ("cases_rate_men5", "Casos · menores de 5"),
    ("cases_rate_60mas", "Casos · 60 y más"),
    ("hosp_rate_men5", "Hospitalizaciones · menores de 5"),
    ("hosp_rate_60mas", "Hospitalizaciones · 60 y más"),
    ("death_rate_men5", "Defunciones · menores de 5"),
    ("death_rate_60mas", "Defunciones · 60 y más"),
]


def _norm(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c)).upper().strip()


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def cargar_predicciones() -> dict[str, dict[str, pd.DataFrame]]:
    """{departamento: {modelo: predicciones}} desde los snapshots de la grilla."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for sub in sorted(GRID.iterdir()):
        met, pred = sub / "metrics.json", sub / "predictions.csv"
        if not (met.exists() and pred.exists()):
            continue
        info = json.loads(met.read_text(encoding="utf-8-sig"))
        dept, modelo = info["department"], info["model"]
        df = pd.read_csv(pred, parse_dates=["date"])
        out.setdefault(dept, {})[modelo] = df
    return out


def banda_empirica(df: pd.DataFrame, horizonte: int = 1, nivel: float = 0.80):
    """
    Banda de incertidumbre empírica a partir de los errores de la validación.

    No es un intervalo de predicción del modelo —los modelos de árboles no lo
    producen— sino la dispersión observada de sus propios errores en el
    walk-forward: los cuantiles del error se suman al pronóstico puntual. Es el
    procedimiento más simple que respeta lo que el modelo realmente erró, y debe
    declararse como tal.
    """
    d = df[df["horizon"] == horizonte]
    if d.empty:
        return None, None
    err = (d["actual"] - d["predicted"]).values
    lo, hi = np.quantile(err, [(1 - nivel) / 2, 1 - (1 - nivel) / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Panel 1 — mapa
# ---------------------------------------------------------------------------

def fig_mapa() -> str:
    rates = pd.read_csv(RATES_CSV)
    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    for f in geo["features"]:
        f["id"] = _norm(f["properties"]["NOMBDEP"])
    rates["key"] = rates["department"].map(_norm)
    anios = sorted(rates["year"].unique())
    ids = [f["id"] for f in geo["features"]]

    def z(col, anio):
        s = rates[rates["year"] == anio].set_index("key")[col]
        return [s.get(i, None) for i in ids]

    traces = []
    for i, (col, label) in enumerate(MEASURES):
        traces.append(go.Choropleth(
            geojson=geo, locations=ids, z=z(col, 2019), zmin=0,
            zmax=float(np.nanpercentile(rates[col].dropna(), 97)) if rates[col].notna().any() else 1,
            colorscale="Reds", marker_line_color="white", marker_line_width=0.5,
            colorbar=dict(title="por 100k", thickness=14),
            visible=(i == 0), name=label,
            hovertemplate="<b>%{location}</b><br>%{z:.1f} por 100k<extra></extra>"))

    fig = go.Figure(traces)
    fig.frames = [go.Frame(name=str(a),
                           data=[go.Choropleth(z=z(c, a)) for c, _ in MEASURES])
                  for a in anios]
    botones = []
    for i, (_, label) in enumerate(MEASURES):
        vis = [j == i for j in range(len(MEASURES))]
        botones.append(dict(label=label, method="update", args=[{"visible": vis}]))
    fig.update_layout(
        updatemenus=[dict(buttons=botones, x=0.01, y=1.12, xanchor="left",
                          bgcolor="white", bordercolor="#ddd")],
        sliders=[dict(active=anios.index(2019) if 2019 in anios else 0,
                      currentvalue=dict(prefix="Año: "), pad=dict(t=40),
                      steps=[dict(method="animate", label=str(a),
                                  args=[[str(a)], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0))])
                             for a in anios])])
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator",
                    bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=56, b=10),
                      paper_bgcolor="white")
    return to_html(fig, include_plotlyjs=False, full_html=False, div_id="mapa",
                   config=CFG_MAP, auto_play=False)


# ---------------------------------------------------------------------------
# Panel 2 — criterio
# ---------------------------------------------------------------------------

def fig_criterio(ev: pd.DataFrame) -> str:
    marca = {"costa": "circle", "sierra": "triangle-up", "selva": "square"}
    fig = go.Figure()
    fig.add_hrect(y0=0.60, y1=0.80, fillcolor=COLOR["SARIMA"], opacity=0.06, line_width=0)
    fig.add_vrect(x0=40, x1=400, fillcolor=COLOR["XGBoost"], opacity=0.06, line_width=0)
    fig.add_vrect(x0=1.5, x1=10, fillcolor=COLOR["SARIMA"], opacity=0.06, line_width=0)
    for modelo in ["SARIMA", "RandomForest", "XGBoost"]:
        sub = ev[ev["recomendado"] == modelo]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["media_casos"], y=sub["fuerza_estacional"], mode="markers+text",
            name=modelo, text=[d.title() for d in sub["departamento"]],
            textposition="top center", textfont=dict(size=9),
            marker=dict(size=13, color=COLOR[modelo],
                        symbol=[marca.get(r, "circle") for r in sub["region"]],
                        line=dict(color="white", width=1.2)),
            customdata=np.stack([sub["regla"], sub["mejor_real"],
                                 sub["dentro"].map({True: "sí", False: "no"})], axis=-1),
            hovertemplate=("<b>%{text}</b><br>%{x:.1f} casos/sem · F_S %{y:.2f}"
                           "<br>regla %{customdata[0]} · mejor real %{customdata[1]}"
                           "<br>dentro del empate: %{customdata[2]}<extra></extra>")))
    fig.update_xaxes(type="log", title="Volumen medio (casos por semana)", gridcolor="#eee")
    fig.update_yaxes(title="Fuerza estacional F<sub>S</sub>", gridcolor="#eee")
    fig.update_layout(height=470, margin=dict(l=60, r=20, t=30, b=50),
                      plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(orientation="h", y=-0.17))
    return to_html(fig, include_plotlyjs=False, full_html=False, div_id="criterio", config=CFG)


# ---------------------------------------------------------------------------
# Panel 3 — modelado por departamento
# ---------------------------------------------------------------------------

def fig_modelado(preds: dict, ev: pd.DataFrame, anios: int = 3) -> tuple[str, dict]:
    depts = sorted(preds)
    rec = ev.set_index("departamento")["recomendado"].to_dict()
    fig = go.Figure()
    traza_de: dict[str, list[int]] = {d: [] for d in depts}
    meta: dict[str, dict] = {}

    for d in depts:
        modelos = preds[d]
        ref = next(iter(modelos.values()))
        corte = ref["date"].max() - pd.DateOffset(years=anios)
        obs = (ref[ref["horizon"] == 1][["date", "actual"]]
               .drop_duplicates("date").sort_values("date"))
        obs = obs[obs["date"] >= corte]
        modelo_rec = rec.get(d, "RandomForest")

        # banda del modelo recomendado
        if modelo_rec in modelos:
            lo, hi = banda_empirica(modelos[modelo_rec])
            p = modelos[modelo_rec]
            p = p[(p["horizon"] == 1) & (p["date"] >= corte)].sort_values("date")
            if lo is not None and not p.empty:
                fig.add_trace(go.Scatter(
                    x=list(p["date"]) + list(p["date"][::-1]),
                    y=list(p["predicted"] + hi) + list((p["predicted"] + lo)[::-1]),
                    fill="toself", fillcolor="rgba(79,129,189,0.16)", line=dict(width=0),
                    name="Banda 80 %", hoverinfo="skip", visible=False, showlegend=True))
                traza_de[d].append(len(fig.data) - 1)

        fig.add_trace(go.Scatter(x=obs["date"], y=obs["actual"], mode="lines",
                                 name="Observado", line=dict(color="#222", width=2),
                                 visible=False))
        traza_de[d].append(len(fig.data) - 1)

        for modelo in CANDIDATOS:
            if modelo not in modelos:
                continue
            p = modelos[modelo]
            p = p[(p["horizon"] == 1) & (p["date"] >= corte)].sort_values("date")
            if p.empty:
                continue
            es_rec = modelo == modelo_rec
            fig.add_trace(go.Scatter(
                x=p["date"], y=p["predicted"], mode="lines",
                name=modelo + (" ★" if es_rec else ""),
                line=dict(color=COLOR[modelo], width=2.2 if es_rec else 1.2,
                          dash="solid" if es_rec else "dot"),
                visible=False))
            traza_de[d].append(len(fig.data) - 1)

        fila = ev[ev["departamento"] == d]
        meta[d] = {
            "recomendado": modelo_rec,
            "regla": str(fila["regla"].iloc[0]) if len(fila) else "—",
            "mejor": str(fila["mejor_real"].iloc[0]) if len(fila) else "—",
            "dentro": bool(fila["dentro"].iloc[0]) if len(fila) else False,
            "emitido": str(ref["date"].max().date()),
            "origenes": int((ref["horizon"] == 1).sum()),
        }

    for i in traza_de[depts[0]]:
        fig.data[i].visible = True

    botones = []
    for d in depts:
        vis = [False] * len(fig.data)
        for i in traza_de[d]:
            vis[i] = True
        botones.append(dict(label=d.title(), method="update",
                            args=[{"visible": vis},
                                  {"title": f"{d.title()} · horizonte 1 semana"}]))
    fig.update_layout(
        updatemenus=[dict(buttons=botones, x=0.01, y=1.16, xanchor="left",
                          bgcolor="white", bordercolor="#ddd")],
        height=430, margin=dict(l=60, r=20, t=70, b=40),
        title=f"{depts[0].title()} · horizonte 1 semana",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Casos semanales",
        legend=dict(orientation="h", y=-0.14))
    return (to_html(fig, include_plotlyjs=False, full_html=False,
                    div_id="modelado", config=CFG), meta)


# ---------------------------------------------------------------------------
# Panel 4 — SHAP
# ---------------------------------------------------------------------------

FAMILIAS = [("inercia_corta", "Inercia corta", "#4f81bd"),
            ("memoria_media", "Memoria media", "#8fb2d8"),
            ("rezago_anual", "Rezago anual", "#e39b97"),
            ("ventanas_moviles", "Ventanas móviles", "#6f9c3d"),
            ("calendario_estacional", "Calendario", "#c0504d"),
            ("tendencia", "Tendencia", "#cfcfcf")]


def fig_shap_global(fam: pd.DataFrame) -> str:
    d = fam[fam["modelo"] == "RandomForest"].sort_values("media_casos")
    etiquetas = [f"{r.departamento.title()} ({r.media_casos:.0f})" for r in d.itertuples()]
    fig = go.Figure()
    for col, nombre, color in FAMILIAS:
        if col not in d.columns:
            continue
        fig.add_trace(go.Bar(y=etiquetas, x=d[col], name=nombre, orientation="h",
                             marker_color=color,
                             hovertemplate="%{y}<br>" + nombre + ": %{x:.1%}<extra></extra>"))
    fig.update_layout(barmode="stack", height=420,
                      margin=dict(l=150, r=20, t=20, b=40),
                      xaxis=dict(title="Fracción de la atribución total",
                                 tickformat=".0%", range=[0, 1]),
                      plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(orientation="h", y=-0.16))
    return to_html(fig, include_plotlyjs=False, full_html=False,
                   div_id="shapglobal", config=CFG)


def fig_shap_local(locales: list[dict]) -> tuple[str, dict]:
    fig = go.Figure()
    traza_de, meta = {}, {}
    for k, loc in enumerate(locales):
        d = loc["departamento"]
        contribs = loc["contribuciones"]
        etiquetas = [f'{c["variable"]} = {c["valor_variable"]:g}' for c in contribs]
        valores = [c["contribucion"] for c in contribs]
        if loc.get("n_variables_restantes"):
            etiquetas.append(f'otras {loc["n_variables_restantes"]} variables')
            valores.append(loc["resto_agrupado"])
        base, acum = loc["valor_base"], loc["valor_base"]
        bases, anchos = [], []
        for v in valores:
            bases.append(acum if v > 0 else acum + v)
            anchos.append(abs(v))
            acum += v
        fig.add_trace(go.Bar(
            y=etiquetas[::-1], x=anchos[::-1], base=bases[::-1], orientation="h",
            marker_color=[COLOR["SARIMA"] if v > 0 else COLOR["RandomForest"]
                          for v in valores][::-1],
            name=d.title(), visible=(k == 0), showlegend=False,
            text=[f"{v:+.2f}" for v in valores][::-1], textposition="outside",
            hovertemplate="%{y}<br>aporta %{text}<extra></extra>"))
        traza_de[d] = k
        meta[d] = {"semana": loc["semana"], "base": base,
                   "prediccion": loc["prediccion"],
                   "eficiencia": loc.get("eficiencia_verificada", False)}

    botones = []
    for d, k in traza_de.items():
        vis = [i == k for i in range(len(fig.data))]
        m = meta[d]
        botones.append(dict(
            label=d.title(), method="update",
            args=[{"visible": vis},
                  {"title": (f"{d.title()} · semana del {m['semana']} · "
                             f"base {m['base']:.1f} → predicción {m['prediccion']:.1f}")}]))
    primero = list(traza_de)[0]
    m0 = meta[primero]
    fig.update_layout(
        updatemenus=[dict(buttons=botones, x=0.01, y=1.18, xanchor="left",
                          bgcolor="white", bordercolor="#ddd")],
        title=(f"{primero.title()} · semana del {m0['semana']} · "
               f"base {m0['base']:.1f} → predicción {m0['prediccion']:.1f}"),
        height=430, margin=dict(l=190, r=60, t=80, b=40),
        xaxis_title="Casos semanales", plot_bgcolor="white", paper_bgcolor="white")
    return (to_html(fig, include_plotlyjs=False, full_html=False,
                    div_id="shaplocal", config=CFG), meta)


# ---------------------------------------------------------------------------
# Tablas HTML
# ---------------------------------------------------------------------------

def tabla(df: pd.DataFrame, clases="") -> str:
    th = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in df.columns)
    filas = []
    for _, r in df.iterrows():
        tds = []
        for v in r:
            if isinstance(v, (bool, np.bool_)):
                tds.append(f'<td class="{"ok" if v else "no"}">{"sí" if v else "no"}</td>')
            elif isinstance(v, (float, np.floating)):
                tds.append(f"<td class='num'>{v:,.2f}</td>".replace(",", " "))
            else:
                tds.append(f"<td>{v}</td>")
        filas.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table class="{clases}"><thead><tr>{th}</tr></thead><tbody>{"".join(filas)}</tbody></table>'


CSS = """
:root{--granate:#841617;--tinta:#1c1c1c;--suave:#5e5e5e;--linea:#e3e3e3;}
*{box-sizing:border-box}
body{margin:0;background:#f3f4f6;color:var(--tinta);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{background:var(--granate);color:#fff;padding:22px 16px 18px;text-align:center}
header h1{margin:0;font-size:25px;letter-spacing:-.2px}
header p{margin:6px 0 0;font-size:13px;opacity:.9}
.badge{display:inline-block;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.35);
 border-radius:11px;padding:1px 9px;font-size:12px;margin-left:8px;vertical-align:middle}
.wrap{max-width:1180px;margin:0 auto;padding:18px 14px 70px}
.kpis{display:flex;gap:11px;flex-wrap:wrap;margin:4px 0 20px}
.kpi{flex:1 1 170px;background:#fff;border:1px solid var(--linea);border-left:4px solid var(--granate);
 border-radius:7px;padding:11px 13px}
.kpi .v{font-size:22px;font-weight:700;line-height:1.15}
.kpi .l{font-size:11px;color:var(--suave);text-transform:uppercase;letter-spacing:.4px;margin-top:2px}
.card{background:#fff;border:1px solid var(--linea);border-radius:9px;padding:18px 20px 20px;margin-bottom:20px}
.card h2{margin:0 0 4px;color:var(--granate);font-size:19px}
.card .sub{margin:0 0 14px;color:var(--suave);font-size:13.5px}
.nota{background:#f7f9fb;border-left:3px solid #9bb4cc;padding:10px 13px;margin:13px 0;font-size:13.5px}
.nota.alerta{background:#fdf6e9;border-left-color:#d9a441}
.nota.bien{background:#f1f7ee;border-left-color:#7aa85c}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:10px}
th,td{border:1px solid var(--linea);padding:5px 8px;text-align:left}
th{background:#eef1f5;font-weight:600}
tr:nth-child(even) td{background:#fafbfc}
td.num{text-align:right;font-variant-numeric:tabular-nums}
td.ok{color:#2f7d32;font-weight:600}
td.no{color:#b3261e;font-weight:600}
.meta{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--suave);margin-top:8px}
.meta b{color:var(--tinta)}
footer{text-align:center;color:var(--suave);font-size:12px;padding:10px 0 30px}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anios", type=int, default=3,
                    help="Años mostrados en el panel de modelado (por defecto 3)")
    ap.add_argument("--salida", default=str(PRUEBAS_DIR / "frontend_neumonia_v2.html"))
    args = ap.parse_args()

    print("Cargando resultados...")
    preds = cargar_predicciones()
    perfiles = pd.read_csv(PROF_DIR / "perfiles_under5.csv")
    datos = json.loads((TESIS_DIR / "datos_capitulos.json").read_text(encoding="utf-8"))
    ev = pd.DataFrame(datos["criterio_en_muestra"]["filas"] +
                      datos["criterio_fuera_muestra"]["filas"])
    fam = pd.read_csv(SHAP_DIR / "atribucion_familias.csv")
    locales = json.loads((SHAP_DIR / "explicaciones_locales.json").read_text(encoding="utf-8"))
    corr = pd.read_csv(SHAP_DIR / "correlacion_perfil.csv")
    estr = pd.DataFrame(datos["criterio_fuera_muestra"]["estrategias"])

    print(f"  {len(preds)} departamentos · {len(locales)} explicaciones locales")

    print("Construyendo figuras...")
    html_mapa = fig_mapa()
    html_crit = fig_criterio(ev)
    html_mod, meta_mod = fig_modelado(preds, ev, anios=args.anios)
    html_sg = fig_shap_global(fam)
    html_sl, meta_sl = fig_shap_local(locales)

    emp = datos["empates"]
    cob_total = sum(1 for r in ev.to_dict("records") if r["dentro"])
    ahorro = datos["criterio_fuera_muestra"]["ahorro_pct"]

    # --- tablas
    t_criterio = ev[["departamento", "region", "media_casos", "fuerza_estacional",
                     "regla", "recomendado", "mejor_real", "dentro", "regret_pct"]].copy()
    t_criterio["departamento"] = t_criterio["departamento"].str.title()
    t_criterio.columns = ["Departamento", "Región", "Casos/sem", "F_S", "Regla",
                          "Recomendado", "Mejor observado", "En el empate", "Exceso %"]

    des = pd.DataFrame(datos["desempeno"])
    t_des = des[["departamento", "SARIMA", "RandomForest", "XGBoost", "LSTM",
                 "mejor", "n_empatados"]].copy()
    t_des["departamento"] = t_des["departamento"].str.title()
    t_des.columns = ["Departamento", "SARIMA", "Random Forest", "XGBoost", "LSTM",
                     "Menor error", "Equivalentes"]

    t_estr = estr.copy()
    t_estr.columns = ["Estrategia", "Cobertura", "Cobertura %", "Exceso medio %", "Exceso máx %"]

    t_corr = corr.copy()
    t_corr["familia"] = t_corr["familia"].str.replace("_", " ")
    t_corr.columns = ["Modelo", "Familia de variables", "r con log(volumen)", "r con F_S"]

    html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pronóstico de Neumonía — Perú · v2.0</title>
<style>{CSS}</style>
<script>{get_plotlyjs()}</script>
</head><body>

<header>
  <h1>Pronóstico de Neumonía — Perú <span class="badge">v2.0</span></h1>
  <p>Vigilancia, modelado y selección automática de modelo · datos IRAS (MINSA) · UNSA</p>
</header>

<div class="wrap">

  <div class="kpis">
    <div class="kpi"><div class="v">{len(preds)}</div><div class="l">Departamentos modelados</div></div>
    <div class="kpi"><div class="v">{datos['configuracion']['n_corridas']}</div><div class="l">Corridas walk-forward</div></div>
    <div class="kpi"><div class="v">{emp['n_con_3_o_mas']}/{emp['n_total']}</div><div class="l">Series con 3+ modelos equivalentes</div></div>
    <div class="kpi"><div class="v">{cob_total}/{len(ev)}</div><div class="l">Cobertura del criterio</div></div>
    <div class="kpi"><div class="v">{ahorro:.0f}%</div><div class="l">Ahorro de cómputo</div></div>
  </div>

  <div class="card">
    <h2>1 · Mapa interactivo</h2>
    <p class="sub">Tasa de incidencia por 100 000 habitantes. Elige la medida y el grupo de edad
    en el menú superior; desplaza el año con el control inferior.</p>
    {html_mapa}
  </div>

  <div class="card">
    <h2>2 · Criterio de selección automática de modelo</h2>
    <p class="sub">Cada departamento ubicado por su <b>perfil estadístico</b>, no por su nombre.
    El color indica la familia que el criterio recomienda; las zonas sombreadas son las regiones
    de decisión de las reglas.</p>
    {html_crit}
    <div class="nota">El criterio decide con dos propiedades del perfil: el volumen medio semanal
    y la fuerza estacional. Los umbrales de 10 y 40 casos por semana se derivan del ruido relativo
    de un conteo de Poisson (1/√λ = 32 % y 16 %), no del ajuste a estos datos.</div>
    {tabla(t_criterio)}
  </div>

  <div class="card">
    <h2>3 · Modelado por departamento</h2>
    <p class="sub">Observado frente a predicho en la validación de origen móvil, últimos
    {args.anios} años. El modelo marcado con ★ es el que recomienda el criterio; su banda
    sombreada es la dispersión empírica al 80 % de sus propios errores.</p>
    {html_mod}
    <div class="meta">
      <span>Horizonte: <b>1 semana</b></span>
      <span>Paso entre orígenes: <b>4 semanas</b></span>
      <span>Ventana de entrenamiento: <b>520 semanas</b></span>
      <span>Última semana con dato: <b>{meta_mod[sorted(preds)[0]]['emitido']}</b></span>
    </div>
    <div class="nota alerta">La banda no es un intervalo de predicción del modelo —los modelos de
    árboles no lo producen— sino el rango central del 80 % de los errores que ese modelo cometió
    en la propia validación. Es una medida de dispersión observada, y así debe interpretarse.</div>
  </div>

  <div class="card">
    <h2>4 · Explicabilidad (SHAP)</h2>
    <p class="sub">La atribución no se usa aquí como adorno interpretativo sino para verificar que
    el modelo se apoya en la estructura que el perfil declara.</p>

    <h3 style="font-size:15px;margin:16px 0 4px">Atribución global por familia de variables</h3>
    <p class="sub">Departamentos ordenados por volumen creciente.</p>
    {html_sg}
    <div class="nota bien">La atribución sigue al <b>volumen</b>, no a la estacionalidad: la
    participación de los rezagos cortos correlaciona con el logaritmo del volumen en
    r = +0,87 (Random Forest) y +0,80 (XGBoost), mientras que frente a la fuerza estacional todas
    las correlaciones son prácticamente nulas. A mayor volumen el modelo abandona los promedios
    suavizados y se apoya en el rezago inmediato, porque deja de estar dominado por el ruido de
    notificación.</div>
    {tabla(t_corr)}

    <h3 style="font-size:15px;margin:22px 0 4px">Descomposición de una predicción concreta</h3>
    <p class="sub">Por qué el modelo pronosticó lo que pronosticó en la semana de mayor incidencia
    de cada departamento. Las contribuciones suman exactamente la diferencia entre el valor base y
    la predicción (propiedad de eficiencia, verificada en cada caso).</p>
    {html_sl}
  </div>

  <div class="card">
    <h2>5 · Comparación de modelos y evaluación del criterio</h2>
    <p class="sub">Error absoluto medio por departamento y familia, horizonte de una semana. La
    última columna es el número de modelos cuyo error no se distingue estadísticamente del mejor.</p>
    {tabla(t_des)}
    <div class="nota alerta">En {emp['n_con_3_o_mas']} de los {emp['n_total']} departamentos hay
    tres o más modelos estadísticamente indistinguibles entre sí. El ordenamiento por valor puntual
    del error describe, en buena parte, variación muestral y no diferencias reales de capacidad
    predictiva.</div>

    <h3 style="font-size:15px;margin:22px 0 4px">El criterio frente a estrategias de modelo fijo</h3>
    <p class="sub">Sobre los 14 departamentos que no intervinieron en la construcción de las reglas.</p>
    {tabla(t_estr)}
    <div class="nota alerta">El criterio de reglas discretas <b>no supera</b> en exactitud a una
    estrategia de modelo fijo. Su aporte está en descartar el aprendizaje profundo antes de
    entrenarlo, en identificar dónde basta el modelo interpretable, y en el ahorro de cómputo.</div>
  </div>

</div>
<footer>Generado el {date.today().isoformat()} · pruebasg/build_frontend_v2.py</footer>
</body></html>"""

    salida = Path(args.salida)
    salida.write_text(html, encoding="utf-8")
    print(f"Escrito: {salida.relative_to(ROOT)}  ({salida.stat().st_size/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
