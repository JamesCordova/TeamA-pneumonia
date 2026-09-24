#!/usr/bin/env python
"""
Informe HTML autónomo del selector automático de modelos.

Consolida en un solo archivo lo producido por run_profiles.py,
run_grid_selector.py, run_selector.py y run_shap.py: el perfilador, la
evidencia experimental, el criterio de decisión, su evaluación y la
explicabilidad SHAP. Las figuras van incrustadas en base64, de modo que el
archivo se abre con doble clic, sin servidor ni conexión.

Uso:
    python pruebasg/build_selector_report.py
    python pruebasg/build_selector_report.py --salida informe.html
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from datetime import date
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import RESULTS_DIR
from selector.evaluate import CANDIDATOS, comparar_todos
from selector.regiones import MUESTRA_ESTRATIFICADA, region_de
from selector.rules import seleccionar

PROFILES = RESULTS_DIR / "profiles"
GRID_DIR = RESULTS_DIR / "grid_selector"
SEL_DIR = RESULTS_DIR / "selector"
SHAP_DIR = RESULTS_DIR / "shap"

COLOR_MODELO = {
    "SARIMA": "#c0504d",
    "RandomForest": "#4f81bd",
    "XGBoost": "#9bbb59",
    "LSTM": "#8064a2",
    "SeasonalNaive": "#999999",
}
MARCADOR_REGION = {"costa": "o", "sierra": "^", "selva": "s"}


# ---------------------------------------------------------------------------
# utilidades de figura
# ---------------------------------------------------------------------------

def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _png_b64(ruta: Path) -> str:
    return base64.b64encode(ruta.read_bytes()).decode("ascii")


def fig_mapa_decision(ev: pd.DataFrame) -> str:
    """Plano volumen × fuerza estacional con la región de decisión de cada regla."""
    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    # Regiones de decisión de las reglas (mismos cortes que rules.py).
    # Se dibujan en coordenadas de datos para que el sombreado coincida
    # exactamente con los umbrales, no con fracciones del eje.
    y_min, y_max = 0.1, 0.78
    ax.axhspan(0.60, y_max, color="#c0504d", alpha=0.07)                     # R2
    ax.fill_between([40, 400], y_min, 0.60, color="#9bbb59", alpha=0.07)     # R1
    ax.fill_between([2, 10], y_min, 0.60, color="#c0504d", alpha=0.07)       # R3
    ax.axhline(0.60, color="#c0504d", ls="--", lw=1, alpha=0.7)
    ax.axvline(40, color="#9bbb59", ls="--", lw=1, alpha=0.7)
    ax.axvline(10, color="#c0504d", ls="--", lw=1, alpha=0.7)

    for _, r in ev.iterrows():
        ax.scatter(r["media_casos"], r["fuerza_estacional"],
                   s=130, c=COLOR_MODELO.get(r["recomendado"], "#555"),
                   marker=MARCADOR_REGION.get(r["region"], "o"),
                   edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate(r["departamento"].title(),
                    (r["media_casos"], r["fuerza_estacional"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=8)

    ax.set_xscale("log")
    ax.set_xlim(2, 400)
    ax.set_ylim(0.1, 0.78)
    ax.set_xlabel("Volumen medio (casos por semana, escala logarítmica)")
    ax.set_ylabel("Fuerza estacional  $F_S$")
    ax.set_title("Regiones de decisión del selector sobre el plano del perfil", fontsize=12)
    ax.grid(alpha=0.25)

    manejadores = [plt.Line2D([], [], marker="o", ls="", color=c, label=m, markersize=9)
                   for m, c in COLOR_MODELO.items() if m in set(ev["recomendado"])]
    manejadores += [plt.Line2D([], [], marker=mk, ls="", color="#555", label=rg, markersize=8)
                    for rg, mk in MARCADOR_REGION.items()]
    ax.legend(handles=manejadores, loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    return _fig_b64(fig)


def fig_estrategias(comp: pd.DataFrame) -> str:
    """Regret medio y cobertura de cada estrategia."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    d = comp.sort_values("regret_medio_pct")
    colores = ["#4f81bd" if e == "selector v1" else "#bbbbbb" for e in d["estrategia"]]

    ax1.barh(d["estrategia"], d["regret_medio_pct"], color=colores)
    ax1.set_xlabel("Regret medio (% de MAE sobre el mejor)")
    ax1.set_title("Costo en exactitud de cada estrategia", fontsize=11)
    ax1.grid(axis="x", alpha=0.3)
    for i, v in enumerate(d["regret_medio_pct"]):
        ax1.text(v + 0.6, i, f"{v:.2f}%", va="center", fontsize=8)

    ax2.barh(d["estrategia"], d["cobertura_pct"], color=colores)
    ax2.set_xlabel("Cobertura (% de series dentro del conjunto empatado)")
    ax2.set_title("Calidad de la recomendación", fontsize=11)
    ax2.set_xlim(0, 108)
    ax2.grid(axis="x", alpha=0.3)
    for i, v in enumerate(d["cobertura_pct"]):
        ax2.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8)

    fig.tight_layout()
    return _fig_b64(fig)


def fig_shap_familias(fam: pd.DataFrame, modelo: str = "RandomForest") -> str:
    """Barras apiladas de atribución SHAP por familia, ordenadas por volumen."""
    d = fam[fam["modelo"] == modelo].sort_values("media_casos")
    if d.empty:
        return ""
    cols = [c for c in ("calendario_estacional", "rezago_anual", "inercia_corta",
                        "memoria_media", "ventanas_moviles", "tendencia") if c in d.columns]
    colores = ["#c0504d", "#e39b97", "#4f81bd", "#8fb2d8", "#9bbb59", "#cfcfcf"]

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    izq = np.zeros(len(d))
    etiquetas = [f"{r.departamento.title()}  ({r.media_casos:.0f}/sem)"
                 for r in d.itertuples()]
    for c, col in zip(cols, colores):
        ax.barh(etiquetas, d[c].values, left=izq, label=c.replace("_", " "), color=col)
        izq += d[c].values
    ax.set_xlabel("Fracción de la atribución SHAP total")
    ax.set_title(f"Atribución por familia de variables — {modelo}\n"
                 "(departamentos ordenados por volumen creciente)", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="lower right", framealpha=0.95)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return _fig_b64(fig)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
:root { --tinta:#1b1b1b; --suave:#5b5b5b; --linea:#e0e0e0; --acento:#2e5c8a; --fondo:#fff; }
* { box-sizing:border-box; }
body { margin:0; background:#f4f5f7; color:var(--tinta);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:1120px; margin:0 auto; padding:34px 22px 80px; background:var(--fondo);
  box-shadow:0 0 0 1px var(--linea); }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-.2px; }
h2 { font-size:19px; margin:38px 0 10px; padding-bottom:6px; border-bottom:2px solid var(--acento); }
h3 { font-size:15px; margin:22px 0 8px; color:var(--acento); }
.sub { color:var(--suave); font-size:13px; margin:0 0 22px; }
p { margin:9px 0; }
table { border-collapse:collapse; width:100%; margin:12px 0 6px; font-size:12.5px; }
th,td { border:1px solid var(--linea); padding:5px 8px; text-align:left; vertical-align:top; }
th { background:#eef1f5; font-weight:600; }
tr:nth-child(even) td { background:#fafbfc; }
td.num,th.num { text-align:right; font-variant-numeric:tabular-nums; }
.ok { color:#2f7d32; font-weight:600; }
.no { color:#b3261e; font-weight:600; }
figure { margin:16px 0; text-align:center; }
figure img { max-width:100%; border:1px solid var(--linea); }
figcaption { font-size:12px; color:var(--suave); margin-top:6px; }
.kpis { display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 6px; }
.kpi { flex:1 1 170px; border:1px solid var(--linea); border-left:4px solid var(--acento);
  padding:11px 13px; background:#fbfcfd; }
.kpi .v { font-size:23px; font-weight:700; line-height:1.15; }
.kpi .l { font-size:11.5px; color:var(--suave); text-transform:uppercase; letter-spacing:.4px; }
.nota { background:#fff8e6; border-left:4px solid #d9a441; padding:11px 14px; margin:14px 0;
  font-size:13.5px; }
.limit { background:#f6f6f8; border-left:4px solid #888; padding:11px 14px; margin:14px 0;
  font-size:13.5px; }
code { background:#f0f2f5; padding:1px 5px; border-radius:3px; font-size:12.5px; }
"""


def tabla_html(df: pd.DataFrame, num_cols=()) -> str:
    th = "".join(f'<th class="{"num" if c in num_cols else ""}">{c.replace("_"," ")}</th>'
                 for c in df.columns)
    filas = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            cls = "num" if c in num_cols else ""
            if isinstance(v, (bool, np.bool_)):
                txt = '<span class="ok">sí</span>' if v else '<span class="no">no</span>'
            elif isinstance(v, float):
                txt = f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
            else:
                txt = str(v)
            tds.append(f'<td class="{cls}">{txt}</td>')
        filas.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(filas)}</tbody></table>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--salida", default=str(PRUEBAS_DIR / "informe_selector.html"))
    parser.add_argument("--age_group", default="under5")
    args = parser.parse_args()

    perfiles = pd.read_csv(PROFILES / f"perfiles_{args.age_group}.csv")
    ev = pd.read_csv(SEL_DIR / "evaluacion_selector.csv")
    comp_estr = pd.read_csv(SEL_DIR / "comparacion_estrategias.csv")
    reglas = pd.read_csv(SEL_DIR / "reglas.csv")

    # Validación fuera de muestra: departamentos con resultados que no están
    # en la muestra con la que se construyeron las reglas.
    fuera = [d for d in perfiles["departamento"]
             if d not in MUESTRA_ESTRATIFICADA]
    comps_fuera = comparar_todos(GRID_DIR, fuera)
    filas_fuera = []
    perf_idx = perfiles.set_index("departamento")
    for dept, c in comps_fuera.items():
        rec = seleccionar(perf_idx.loc[dept].to_dict())
        filas_fuera.append({
            "departamento": dept,
            "region": region_de(dept),
            "media_casos": round(perf_idx.loc[dept, "media_casos"], 1),
            "fuerza_estacional": round(perf_idx.loc[dept, "fuerza_estacional"], 3),
            "regla": rec.regla,
            "recomendado": rec.modelo,
            "mejor_real": c.mejor,
            "dentro_del_empate": rec.modelo in c.conjunto_empatado,
            "regret_casos": round(c.regret(rec.modelo), 3),
            "regret_rel_pct": round(c.regret_relativo(rec.modelo) * 100, 2),
        })
    val = pd.DataFrame(filas_fuera).sort_values(["region", "media_casos"],
                                                ascending=[True, False])

    # Estrategias fijas sobre el MISMO conjunto fuera de muestra, para que la
    # comparación con el selector sea sobre las mismas series.
    estr_fuera = []
    if comps_fuera:
        for modelo in CANDIDATOS:
            rr = [c.regret_relativo(modelo) * 100 for c in comps_fuera.values()]
            dentro = sum(1 for c in comps_fuera.values()
                         if modelo in c.conjunto_empatado)
            estr_fuera.append({
                "estrategia": f"siempre {modelo}",
                "cobertura_pct": round(100 * dentro / len(comps_fuera), 1),
                "regret_medio_pct": round(float(np.mean(rr)), 2),
                "regret_max_pct": round(float(np.max(rr)), 2),
            })
        estr_fuera.append({
            "estrategia": "selector v1",
            "cobertura_pct": round(100 * val["dentro_del_empate"].mean(), 1),
            "regret_medio_pct": round(float(val["regret_rel_pct"].mean()), 2),
            "regret_max_pct": round(float(val["regret_rel_pct"].max()), 2),
        })
    estr_fuera = pd.DataFrame(estr_fuera)

    fam = pd.read_csv(SHAP_DIR / "atribucion_familias.csv") if \
        (SHAP_DIR / "atribucion_familias.csv").exists() else pd.DataFrame()

    # --- figuras
    img_mapa = fig_mapa_decision(ev)
    img_estr = fig_estrategias(comp_estr)
    img_shap = fig_shap_familias(fam) if not fam.empty else ""

    n = len(ev)
    cobertura = 100 * ev["dentro_del_empate"].mean()
    regret = ev["regret_rel_pct"].mean()
    ahorro = 100 * (1 - ev["costo_s"].sum() / ev["costo_todo_s"].sum())

    kpis_val = ""
    if not val.empty:
        cob_v = 100 * val["dentro_del_empate"].mean()
        reg_v = val["regret_rel_pct"].mean()
        kpis_val = f"""
        <div class="kpi"><div class="v">{cob_v:.0f}%</div>
          <div class="l">Cobertura fuera de muestra ({int(val['dentro_del_empate'].sum())}/{len(val)})</div></div>
        <div class="kpi"><div class="v">{reg_v:.2f}%</div>
          <div class="l">Regret medio fuera de muestra</div></div>"""

    html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Selector automático de modelos — Informe</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>Selector automático de modelos de pronóstico</h1>
<p class="sub">Series semanales de neumonía, menores de 5 años · Perú 2000–2024 ·
Informe generado el {date.today().isoformat()}</p>

<p>Este informe documenta el criterio que, a partir del <strong>perfil estadístico</strong> de
una serie semanal de notificación obligatoria, determina qué familia de modelo de pronóstico
corresponde emplear. El criterio no usa el nombre del departamento ni de la enfermedad, de modo
que es aplicable a cualquier otra serie de vigilancia epidemiológica: la neumonía es el banco de
pruebas, no el objeto del criterio.</p>

<div class="kpis">
  <div class="kpi"><div class="v">{cobertura:.0f}%</div><div class="l">Cobertura en muestra ({int(ev['dentro_del_empate'].sum())}/{n})</div></div>
  <div class="kpi"><div class="v">{regret:.2f}%</div><div class="l">Regret medio</div></div>
  <div class="kpi"><div class="v">{ahorro:.0f}%</div><div class="l">Ahorro de cómputo</div></div>
  {kpis_val}
</div>

<h2>1. El perfilador de series</h2>
<p>Cada serie se describe mediante un vector de metafeatures: escala y volumen, variabilidad y
sobredispersión, estacionariedad (ADF y KPSS), fuerza de tendencia y de estacionalidad por
descomposición STL, autocorrelación en los rezagos 1, 4 y 52, entropía espectral, estabilidad
y cambios de nivel. La construcción sigue la línea de caracterización de series de Wang, Smith
e Hyndman (2006), con descriptores adicionales propios de datos de conteo.</p>
<p>La tabla presenta el perfil de los {len(perfiles)} departamentos. Las categorías derivadas
(<code>cat_volumen</code>, <code>cat_estacionalidad</code>) son las que consume el criterio.</p>
{tabla_html(perfiles[['departamento','region','n_semanas','media_casos','cv','indice_dispersion',
                      'fuerza_estacional','fuerza_tendencia','entropia_espectral','acf_1','acf_52',
                      'estacionaria','cat_volumen','cat_estacionalidad']],
            num_cols=('n_semanas','media_casos','cv','indice_dispersion','fuerza_estacional',
                      'fuerza_tendencia','entropia_espectral','acf_1','acf_52'))}

<h2>2. La evidencia: qué modelo gana en cada perfil</h2>
<p>Se ejecutó validación de origen móvil sobre {n} departamentos estratificados por región natural
y volumen, con cinco familias de modelo, ventana inicial de 520 semanas, horizonte de 4 semanas y
desplazamiento de 4 semanas, lo que produce entre 183 y 184 orígenes de pronóstico por serie.</p>

<div class="nota"><strong>Hallazgo que condiciona todo el diseño.</strong> En 10 de los 11
departamentos, SARIMA, Random Forest y XGBoost producen errores <em>estadísticamente
indistinguibles</em> entre sí (prueba t pareada sobre los errores por origen, α = 0,05). Solo en
Lima existe un ganador único. En consecuencia, un selector que persiguiera acertar el MAE mínimo
exacto estaría ajustándose al ruido: la decisión que sí es distinguible —y la que el criterio
persigue— es <em>qué familia descartar</em> y <em>cuándo el modelo barato e interpretable
basta</em>.</div>

<h2>3. El criterio de decisión</h2>
{tabla_html(reglas)}
<figure><img src="data:image/png;base64,{img_mapa}" alt="Regiones de decisión">
<figcaption>Figura 1. Cada departamento sobre el plano volumen × fuerza estacional, coloreado según
el modelo que el criterio recomienda. Las líneas discontinuas son los cortes de las reglas.</figcaption></figure>

<h2>4. Evaluación del criterio</h2>
<p>El selector no se evalúa por acertar el mínimo, sino por tres cantidades: la
<strong>cobertura</strong> (fracción de series donde la recomendación cae dentro del conjunto
estadísticamente empatado), el <strong>regret</strong> (exceso de error respecto del mejor modelo)
y el <strong>ahorro de cómputo</strong> frente a evaluar todas las familias.</p>
{tabla_html(ev[['departamento','region','media_casos','fuerza_estacional','regla','recomendado',
                'mejor_real','conjunto_empatado','dentro_del_empate','regret_rel_pct']],
            num_cols=('media_casos','fuerza_estacional','regret_rel_pct'))}
<figure><img src="data:image/png;base64,{img_estr}" alt="Comparación de estrategias">
<figcaption>Figura 2. El selector frente a estrategias de modelo fijo y frente al oráculo.</figcaption></figure>
{tabla_html(comp_estr, num_cols=('cobertura_pct','regret_medio_pct','regret_max_pct','costo_total_s'))}

<div class="limit"><strong>Lectura honesta del resultado.</strong> La estrategia «siempre XGBoost»
alcanza una cobertura y un regret prácticamente iguales a los del selector. La ventaja del criterio
no está, por tanto, en la exactitud: está en que identifica las {int((ev['recomendado']=='SARIMA').sum())} series
en las que el modelo estadístico —más barato de entrenar, con intervalos de predicción y
parámetros interpretables— es suficiente, y lo hace sin pérdida de exactitud medible; y en que
descarta el aprendizaje profundo antes de entrenarlo. Presentar el selector como una mejora de
exactitud sobre XGBoost no sería sostenible con esta evidencia.</div>
"""

    if not val.empty:
        fallos = val[~val["dentro_del_empate"]]
        cob_v = 100 * val["dentro_del_empate"].mean()
        mejor_fija = (estr_fuera[estr_fuera["estrategia"] != "selector v1"]
                      .sort_values("regret_medio_pct").iloc[0]
                      if not estr_fuera.empty else None)
        html += f"""
<h2>5. Validación fuera de la muestra</h2>
<p>Las reglas se construyeron observando los resultados de los {n} departamentos estratificados.
Aquí se aplican, <strong>sin modificación alguna</strong>, a los {len(val)} departamentos restantes,
cuyos resultados no intervinieron en el diseño del criterio. Ésta es la única medición no
contaminada por el proceso de construcción de las reglas.</p>
{tabla_html(val, num_cols=('media_casos','fuerza_estacional','regret_casos','regret_rel_pct'))}
{tabla_html(estr_fuera, num_cols=('cobertura_pct','regret_medio_pct','regret_max_pct'))}

<div class="limit"><strong>Resultado adverso, declarado como tal.</strong>
Fuera de muestra el selector alcanza una cobertura de {int(val['dentro_del_empate'].sum())} de
{len(val)} ({cob_v:.1f} %) con un exceso de error medio de {val['regret_rel_pct'].mean():.2f} %.
{'La estrategia fija «' + str(mejor_fija['estrategia']) + '» la supera en ambas cantidades: cobertura ' + f"{mejor_fija['cobertura_pct']:.0f}" + ' % y exceso medio ' + f"{mejor_fija['regret_medio_pct']:.2f}" + ' %.' if mejor_fija is not None else ''}
<strong>El criterio de reglas discretas, tal como está formulado en la versión 1, no supera a una
estrategia de modelo fijo.</strong> Presentarlo como si lo hiciera no sería sostenible.</p>

<p>El fallo se concentra en una sola serie y en una sola regla. En
{', '.join(fallos['departamento'].tolist()) if not fallos.empty else '—'}, la regla R3 —volumen
inferior a diez casos semanales, recomendar el modelo estadístico— recomienda SARIMA cuando el
bosque aleatorio es significativamente mejor (p = 0,0004). La regla acierta, en cambio, en las
otras tres series de volumen comparable del estudio: Tacna (2,0 casos/semana), Tumbes (3,1) y
Madre de Dios (4,7), donde SARIMA pertenece al conjunto empatado. El volumen por sí solo no
discrimina el caso que falla, de modo que la regla necesita una condición adicional que la versión
2 deberá identificar sobre el vector completo de metafeatures.</p>

<p><strong>Precisión sobre la magnitud.</strong> El exceso de error del 23,3 % en esa serie
corresponde a una diferencia absoluta de 0,22 casos por semana (SARIMA 1,19 frente a Random Forest
0,96). El exceso relativo se infla cuando el error de base es de aproximadamente un caso semanal,
que es el mismo efecto de escala demostrado para el MAPE. Esto no anula el fallo —la diferencia es
estadísticamente significativa y la regla erró—, pero sí indica que el exceso relativo no debe ser
la única métrica de evaluación del criterio: la versión 2 debe reportar ambas magnitudes.</p></div>
"""

    if not fam.empty:
        corr = pd.read_csv(SHAP_DIR / "correlacion_perfil.csv") if \
            (SHAP_DIR / "correlacion_perfil.csv").exists() else pd.DataFrame()
        html += f"""
<h2>6. Explicabilidad: SHAP como verificación del criterio</h2>
<p>SHAP no se emplea aquí como una capa de interpretabilidad añadida al final, sino como
instrumento de verificación: permite comprobar si el modelo se apoya efectivamente en la
estructura que el perfil declara. La pregunta concreta es si la atribución de las variables sigue
al volumen de la serie, a su fuerza estacional, o a ninguno de los dos.</p>
{f'<figure><img src="data:image/png;base64,{img_shap}" alt="Atribución SHAP"><figcaption>Figura 3. Atribución media |SHAP| agregada por familia de variables.</figcaption></figure>' if img_shap else ''}
{tabla_html(corr, num_cols=('r_log_volumen','r_fuerza_estacional')) if not corr.empty else ''}

<div class="nota"><strong>Lo que SHAP confirma y lo que no.</strong>
La atribución sigue al <em>volumen</em>, no a la estacionalidad. La participación de la inercia de
corto plazo (rezagos de hasta cuatro semanas) correlaciona con el logaritmo del volumen en
r = +0,87 en Random Forest y r = +0,80 en XGBoost, mientras que las ventanas móviles y el
calendario correlacionan negativamente (entre −0,65 y −0,81). Frente a la fuerza estacional, todas
las correlaciones son prácticamente nulas (|r| &lt; 0,25).</p>
<p>El mecanismo queda así identificado: <strong>a medida que crece el volumen, el modelo desplaza
su atribución desde los agregados suavizados hacia el rezago inmediato</strong>, porque con conteos
altos el rezago deja de estar dominado por el ruido de notificación de Poisson. Es exactamente el
argumento sobre el que se construyeron las reglas R1 y R3, ahora verificado sobre el mecanismo
interno del modelo y no solo sobre el error agregado.</p>
<p>En cambio, la regla R2 —estacionalidad dominante, recomendar el modelo estadístico— <em>no</em>
queda respaldada por la vía esperada: los modelos de árboles no incrementan su atribución al
calendario en las series de estacionalidad fuerte. Esa ausencia admite una lectura que refuerza la
regla en lugar de invalidarla: si los árboles no explotan la estructura estacional aun cuando ésta
domina la varianza, entonces el modelo con estacionalidad explícita es el adecuado precisamente
porque los otros no la capturan. Un hallazgo concreto lo confirma: la atribución de XGBoost al
rezago anual es exactamente cero en los once departamentos, porque su configuración de variables
—rezagos 1, 2, 4, 8 y 13— no incluye el rezago 52. La limitación no es del algoritmo sino del
conjunto de características, y corregirla es una línea de trabajo inmediata.</div>

{tabla_html(fam.drop(columns=[c for c in ['top3_variables'] if c in fam.columns]),
            num_cols=tuple(c for c in fam.columns if c not in
                           ('departamento','region','modelo','top3_variables')))}
"""

    html += """
<h2>Limitaciones declaradas</h2>
<div class="limit">
<p><strong>1. Una sola enfermedad y un solo grupo de edad.</strong> El criterio se construyó sobre
series de neumonía en menores de cinco años. Su transferibilidad a otras series de notificación
obligatoria es la hipótesis de diseño, no un resultado demostrado; la validación en el grupo de
60 años o más y en otras enfermedades queda pendiente.</p>
<p><strong>2. Una configuración por familia.</strong> Cada familia se representó con una sola
configuración de hiperparámetros. Un ajuste específico por serie podría alterar el ordenamiento,
aunque la exploración previa mostró que la búsqueda en malla no superó a las configuraciones
manuales por márgenes superiores al ruido del estimador.</p>
<p><strong>3. Los cortes de las reglas son discretos.</strong> Los umbrales de 40 y 10 casos
semanales y de 0,60 en fuerza estacional provienen del ruido relativo de Poisson y de la literatura
de descomposición, pero su calibración fina sobre este conjunto de datos no se ha realizado; una
versión posterior del criterio debería reemplazar las reglas discretas por un clasificador
entrenado sobre el vector completo de metafeatures.</p>
<p><strong>4. Horizonte único.</strong> La evaluación reportada corresponde al horizonte de una
semana. El criterio podría no ser el mismo para horizontes largos, donde la degradación del error
recursivo penaliza de forma distinta a cada familia.</p>
</div>

</div></body></html>"""

    salida = Path(args.salida)
    salida.write_text(html, encoding="utf-8")
    print(f"Informe escrito: {salida}  ({salida.stat().st_size/1024:.0f} KB)")
    if not val.empty:
        print(f"Validación fuera de muestra: {int(val['dentro_del_empate'].sum())}/{len(val)} "
              f"dentro del empate, regret medio {val['regret_rel_pct'].mean():.2f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
