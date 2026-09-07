#!/usr/bin/env python
r"""
Frontend web estilo PanViz para el pronóstico de neumonía (Perú).

Genera pruebasg/frontend_neumonia.html: página autónoma (Plotly embebido, sin
internet) con la estética de PanViz (cabecera granate, paneles en tarjetas):

  Panel 1 — Mapa interactivo del Perú por departamento: coropleto de la tasa de
            incidencia por 100k, con selector de medida/grupo de edad y control
            deslizante de año.
  Panel 2 — Modelado (observado vs. predicho) por departamento: backtest de cada
            modelo + proyección futura, con leyenda on/off; comparación por
            horizonte y mapa de calor del error.

Uso:
    python pruebasg/build_frontend.py
"""

from __future__ import annotations

import json
import sys
import unicodedata

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.offline import get_plotlyjs

from _common import PRUEBAS_DIR, ROOT
import build_interactive as bi  # reutiliza collect() y las figuras de modelado

MAROON = "#841617"
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}
# El mapa no debe capturar la rueda del ratón (si no, el scroll de la página lo hace zoom)
MAP_CONFIG = {"displaylogo": False, "responsive": True, "scrollZoom": False}
RATES_CSV = ROOT / "reports" / "tables" / "departmental_annual_variation_rates.csv"
GEOJSON = PRUEBAS_DIR / "peru_departamentos.geojson"

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


def build_map() -> str:
    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    names = [f["properties"]["NOMBDEP"] for f in geo["features"]]
    rates = pd.read_csv(RATES_CSV)
    years = sorted(int(y) for y in rates["year"].unique())
    default_year = 2019 if 2019 in years else years[-1]

    def zvals(col, year):
        sub = rates[rates["year"] == year]
        lut = {_norm(d): v for d, v in zip(sub["department"], sub[col])}
        return [lut.get(_norm(n)) for n in names]

    traces = []
    for i, (col, lab) in enumerate(MEASURES):
        traces.append(go.Choropleth(
            geojson=geo, locations=names, z=zvals(col, default_year),
            featureidkey="properties.NOMBDEP", colorscale="Reds",
            marker_line_color="white", marker_line_width=0.4,
            colorbar=dict(title="por 100k"), name=lab, visible=(i == 0),
            hovertemplate="<b>%{location}</b><br>" + lab + ": %{z:.1f} / 100k<extra></extra>"))
    fig = go.Figure(traces)

    fig.frames = [
        go.Frame(name=str(y),
                 data=[go.Choropleth(z=zvals(col, y)) for col, _ in MEASURES],
                 traces=list(range(len(MEASURES))))
        for y in years
    ]

    steps = [dict(method="animate", label=str(y),
                  args=[[str(y)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for y in years]
    sliders = [dict(active=years.index(default_year), pad=dict(t=50, b=10),
                    currentvalue=dict(prefix="Año: ", font=dict(size=14)), steps=steps)]

    buttons = []
    for i, (col, lab) in enumerate(MEASURES):
        buttons.append(dict(label=lab, method="update",
                            args=[{"visible": [j == i for j in range(len(MEASURES))]}]))
    updatemenus = [dict(buttons=buttons, x=0.0, y=1.09, xanchor="left", yanchor="bottom",
                        direction="down", showactive=True, bgcolor="#f8fafc",
                        bordercolor="#cbd5e1", font=dict(size=12))]

    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator",
                    bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=52, b=10),
                      sliders=sliders, updatemenus=updatemenus,
                      paper_bgcolor="rgba(0,0,0,0)")
    return to_html(fig, include_plotlyjs=False, full_html=False,
                   div_id="mapa", config=MAP_CONFIG, auto_play=False)


def build() -> str:
    map_html = build_map()

    rows = bi.collect()
    depts = sorted({r["department"] for r in rows})
    modeling = ""
    n = 0
    for dept in depts:
        drows = [r for r in rows if r["department"] == dept]
        colors = {r["label"]: bi.PALETTE[i % len(bi.PALETTE)] for i, r in enumerate(drows)}
        ts = to_html(bi.fig_timeseries(drows, colors), include_plotlyjs=False,
                     full_html=False, div_id=f"ts{n}", config=PLOTLY_CONFIG)
        mh = to_html(bi.fig_metric_by_horizon(drows, colors), include_plotlyjs=False,
                     full_html=False, div_id=f"mh{n}", config=PLOTLY_CONFIG)
        hm_fig = bi.fig_heatmap(drows)
        hm = (to_html(hm_fig, include_plotlyjs=False, full_html=False,
                      div_id=f"hm{n}", config=PLOTLY_CONFIG) if hm_fig else "")
        n += 1
        modeling += f"""
        <div class="subpanel">
          <h3>{dept} · menores de 5 años</h3>
          {bi.best_banner(drows)}
          {bi.comparison_table(drows)}
          <div class="cap">Serie de tiempo — clic en la leyenda para aislar modelos; la línea punteada es la proyección futura.</div>
          {ts}
          <div class="cap">Error por horizonte (botones para cambiar la métrica).</div>
          {mh}
          <div class="cap">Mapa de calor del error (MAE) por año — rojo peor, verde mejor.</div>
          {hm}
        </div>"""

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pronóstico de Neumonía — Perú</title>
<script type="text/javascript">{get_plotlyjs()}</script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#f1efe9; color:#1f2937;
         font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  header {{ background:{MAROON}; color:#fff; padding:20px 28px; text-align:center; }}
  header h1 {{ margin:0; font-size:24px; letter-spacing:.3px; }}
  header p {{ margin:6px 0 0; color:#f3d6d6; font-size:13px; }}
  main {{ max-width:1080px; margin:0 auto; padding:20px 16px 60px; }}
  .panel {{ background:#fff; border:1px solid #e2ddd4; border-radius:14px;
           padding:14px 18px 18px; margin-bottom:26px; box-shadow:0 1px 3px rgba(0,0,0,.06); }}
  .panel > h2 {{ text-align:center; color:{MAROON}; font-size:20px; margin:6px 0 14px;
                border-bottom:2px solid #efe6d8; padding-bottom:8px; }}
  .subpanel {{ border-top:1px dashed #e5e0d6; padding-top:12px; margin-top:14px; }}
  .subpanel:first-child {{ border-top:none; margin-top:0; }}
  .subpanel > h3 {{ color:{MAROON}; font-size:16px; margin:6px 0; }}
  .cap {{ font-size:12px; color:#6b7280; margin:14px 0 2px; }}
  .summary {{ background:#fbeaea; border:1px solid #f0c6c6; padding:8px 12px;
             border-radius:8px; font-size:14px; }}
  table.cmp {{ border-collapse:collapse; width:100%; font-size:13px; margin-top:4px; }}
  table.cmp th, table.cmp td {{ border:1px solid #e5e7eb; padding:5px 8px; text-align:right; }}
  table.cmp th {{ background:#faf6f0; }}
  table.cmp td.mcol {{ text-align:left; font-weight:600; background:#faf6f0; }}
  table.cmp td.best {{ background:#dcfce7; font-weight:700; }}
  .note {{ font-size:12px; color:#6b7280; text-align:center; margin-top:8px; }}
</style></head>
<body>
<header>
  <h1>Pronóstico de Neumonía — Perú</h1>
  <p>Vigilancia y modelado de casos por departamento · datos IRAS (MINSA) · UNSA</p>
</header>
<main>
  <div class="panel">
    <h2>Mapa interactivo — tasa por 100 000 habitantes</h2>
    {map_html}
    <div class="note">Elige la medida y el grupo de edad en el menú superior; desplaza el año con el control inferior.</div>
  </div>
  <div class="panel">
    <h2>Modelado — observado vs. predicho</h2>
    {modeling}
  </div>
</main>
</body></html>"""


def main() -> int:
    out = PRUEBAS_DIR / "frontend_neumonia.html"
    out.write_text(build(), encoding="utf-8")
    print(f"✓ Frontend generado: {out}")
    print(f"  {out.stat().st_size/1024/1024:.2f} MB · ábrelo con doble clic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
