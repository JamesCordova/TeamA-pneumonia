#!/usr/bin/env python
r"""
Dashboard interactivo (Recomendaciones 2 y 3 — Gopi).

Genera pruebasg/dashboard_interactive.html: un único archivo autónomo (Plotly.js
embebido, funciona sin internet) con, por departamento:

  - Tabla comparativa de todos los modelos (métricas a h=1 y h=4) lado a lado.
  - Serie de tiempo interactiva: backtest de cada modelo + proyección FUTURA
    punteada (4 semanas más allá del último dato). Leyenda on/off por modelo.
  - Líneas de métrica por horizonte (todos los modelos), con selector de métrica.
  - Mapa de calor modelos × año del error (MAE): rojo=alto, verde=bajo.
  - Lista desplegable de los comandos exactos ejecutados.

Uso:
    python pruebasg/build_interactive.py
"""

from __future__ import annotations

import html
import sys
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.offline import get_plotlyjs

from _common import METRIC_HEADERS, PRUEBAS_DIR, RESULTS_DIR, load_json, load_manifest

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6",
           "#0891b2", "#db2777", "#65a30d"]
METRIC_KEYS = ["mae", "rmse", "smape", "mda"]
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def collect() -> list[dict]:
    rows = []
    for e in load_manifest():
        if not e.get("ok"):
            continue
        by_h, bt, fut, snap = {}, None, None, None
        if e.get("metrics_file"):
            mpath = RESULTS_DIR / e["metrics_file"]
            snap = mpath.parent
            mj = load_json(mpath)
            by_h = {int(h): v for h, v in mj.get("metrics_by_horizon", {}).items()}
        if e.get("pred_file"):
            p = RESULTS_DIR / e["pred_file"]
            if p.exists():
                bt = pd.read_csv(p, parse_dates=["date"])
                bt = bt[bt["split"] == "backtest"].sort_values("date")
        if snap is not None:
            fpath = snap / "future.csv"
            if fpath.exists():
                fut = pd.read_csv(fpath, parse_dates=["date"]).sort_values("date")
        rows.append({
            "label": e["label"], "department": e["department"], "model": e["model"],
            "command": e["command"], "by_h": by_h, "bt": bt, "fut": fut,
        })
    return rows


# ---------------------------------------------------------------------------
# Figuras Plotly
# ---------------------------------------------------------------------------

def fig_timeseries(rows: list[dict], colors: dict) -> go.Figure:
    fig = go.Figure()

    # Línea "Real" (única, tomada de la unión de backtests)
    actual = None
    for r in rows:
        if r["bt"] is not None and not r["bt"].empty:
            a = r["bt"][["date", "actual"]]
            actual = a if actual is None else pd.concat([actual, a])
    if actual is not None:
        actual = actual.dropna().drop_duplicates("date").sort_values("date")
        fig.add_trace(go.Scatter(
            x=actual["date"], y=actual["actual"], name="Real",
            line=dict(color="#111827", width=1.4), legendgroup="real"))
        last_date, last_val = actual["date"].iloc[-1], actual["actual"].iloc[-1]
    else:
        last_date = last_val = None

    for r in rows:
        c = colors[r["label"]]
        base_vis = "legendonly" if "baseline" in r["label"].lower() else True
        if r["bt"] is not None and not r["bt"].empty:
            fig.add_trace(go.Scatter(
                x=r["bt"]["date"], y=r["bt"]["predicted"], name=r["label"],
                legendgroup=r["label"], visible=base_vis,
                line=dict(color=c, width=1.3)))
        if r["fut"] is not None and not r["fut"].empty and last_date is not None:
            fx = [last_date] + list(r["fut"]["date"])
            fy = [last_val] + list(r["fut"]["predicted"])
            fig.add_trace(go.Scatter(
                x=fx, y=fy, name=r["label"] + " (futuro)", legendgroup=r["label"],
                visible=base_vis, showlegend=False,
                line=dict(color=c, width=2.4, dash="dot"),
                marker=dict(size=5), mode="lines+markers"))

    if last_date is not None:
        fig.add_vline(x=last_date, line=dict(color="#94a3b8", width=1, dash="dash"))
        fig.add_annotation(x=last_date, yref="paper", y=1.02, showarrow=False,
                           text="← histórico | futuro →", font=dict(size=10, color="#64748b"))

    fig.update_layout(
        hovermode="x unified", height=460, margin=dict(l=50, r=20, t=34, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, x=0),
        yaxis_title="Casos")
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.06)
    return fig


def fig_metric_by_horizon(rows: list[dict], colors: dict) -> go.Figure:
    fig = go.Figure()
    horizons = sorted({h for r in rows for h in r["by_h"]})
    trace_metric = []
    for m in METRIC_KEYS:
        for r in rows:
            ys = [r["by_h"].get(h, {}).get(m) for h in horizons]
            fig.add_trace(go.Scatter(
                x=horizons, y=ys, name=r["label"], legendgroup=r["label"],
                mode="lines+markers", line=dict(color=colors[r["label"]], width=2),
                visible=(m == "mae")))
            trace_metric.append(m)

    buttons = []
    for m in METRIC_KEYS:
        vis = [tm == m for tm in trace_metric]
        buttons.append(dict(label=METRIC_HEADERS[m], method="update",
                            args=[{"visible": vis},
                                  {"yaxis": {"title": METRIC_HEADERS[m]}}]))

    fig.update_layout(
        height=440, margin=dict(l=50, r=20, t=44, b=40),
        xaxis=dict(title="Horizonte (semanas adelante)", tickmode="array", tickvals=horizons),
        yaxis_title=METRIC_HEADERS["mae"],
        updatemenus=[dict(type="buttons", buttons=buttons, direction="right",
                          x=0, xanchor="left", y=1.0, yanchor="bottom",
                          showactive=True, bgcolor="#f8fafc", bordercolor="#cbd5e1",
                          font=dict(size=11), pad=dict(t=1, b=1, l=1, r=1))])
    return fig


def fig_heatmap(rows: list[dict]) -> go.Figure | None:
    series = {}
    all_years = set()
    for r in rows:
        if r["bt"] is None or r["bt"].empty:
            continue
        g = r["bt"].dropna(subset=["actual", "predicted"]).copy()
        g["year"] = g["date"].dt.year
        g["ae"] = (g["actual"] - g["predicted"]).abs()
        by_year = g.groupby("year")["ae"].mean()
        series[r["label"]] = by_year
        all_years.update(by_year.index.tolist())
    if not series:
        return None

    years = sorted(all_years)
    labels = list(series.keys())
    z = [[round(float(series[lbl].get(y, float("nan"))), 1) for y in years] for lbl in labels]

    fig = go.Figure(go.Heatmap(
        z=z, x=[str(y) for y in years], y=labels,
        colorscale="RdYlGn", reversescale=True, colorbar=dict(title="MAE"),
        text=z, texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="%{y}<br>%{x}: MAE=%{z}<extra></extra>"))
    fig.update_layout(
        height=70 + 34 * len(labels), margin=dict(l=180, r=20, t=20, b=40))
    return fig


# ---------------------------------------------------------------------------
# Tabla y HTML
# ---------------------------------------------------------------------------

def comparison_table(rows: list[dict]) -> str:
    horizons = sorted({h for r in rows for h in r["by_h"]})
    if not horizons:
        return ""
    h_short, h_long = horizons[0], horizons[-1]
    cols = [(h_short, "mae"), (h_short, "rmse"), (h_short, "smape"),
            (h_short, "mda"), (h_long, "mae"), (h_long, "rmse")]

    # mejores por columna (min salvo MDA que es max)
    best = {}
    for h, k in cols:
        vals = [r["by_h"].get(h, {}).get(k) for r in rows if r["by_h"].get(h, {}).get(k) is not None]
        if vals:
            best[(h, k)] = max(vals) if k == "mda" else min(vals)

    head = "".join(f"<th>h{h} {METRIC_HEADERS[k]}</th>" for h, k in cols)
    body = ""
    order = sorted(rows, key=lambda r: r["by_h"].get(h_short, {}).get("mae", float("inf")))
    for r in order:
        cells = ""
        for h, k in cols:
            v = r["by_h"].get(h, {}).get(k)
            txt = "—" if v is None else f"{v:.2f}"
            is_best = v is not None and (h, k) in best and abs(v - best[(h, k)]) < 1e-9
            cells += f"<td class='{'best' if is_best else ''}'>{txt}</td>"
        body += f"<tr><td class='mcol'>{html.escape(r['label'])}</td>{cells}</tr>"
    return (f"<table class='cmp'><thead><tr><th>Modelo</th>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table>")


def commands_block(rows: list[dict]) -> str:
    items = "".join(
        f"<div class='cmdrow'><b>{html.escape(r['label'])}</b>"
        f"<code>{html.escape(r['command'])}</code></div>" for r in rows)
    return (f"<details><summary>Ver los comandos exactos ejecutados "
            f"({len(rows)} modelos)</summary>{items}</details>")


def best_banner(rows: list[dict]) -> str:
    scored = [(r["label"], r["by_h"].get(min(r["by_h"]), {}).get("mae"))
              for r in rows if r["by_h"]]
    scored = [(l, v) for l, v in scored if v is not None]
    if not scored:
        return ""
    lbl, v = min(scored, key=lambda t: t[1])
    return (f"<p class='summary'>🏆 Mejor a corto plazo (MAE h=1): "
            f"<b>{html.escape(lbl)}</b> ({v:.2f} casos)</p>")


def build(rows: list[dict]) -> str:
    depts = sorted({r["department"] for r in rows})
    sections = ""
    n = 0
    for dept in depts:
        drows = [r for r in rows if r["department"] == dept]
        colors = {r["label"]: PALETTE[i % len(PALETTE)] for i, r in enumerate(drows)}

        ts = to_html(fig_timeseries(drows, colors), include_plotlyjs=False,
                     full_html=False, div_id=f"ts{n}", config=PLOTLY_CONFIG)
        mh = to_html(fig_metric_by_horizon(drows, colors), include_plotlyjs=False,
                     full_html=False, div_id=f"mh{n}", config=PLOTLY_CONFIG)
        hm_fig = fig_heatmap(drows)
        hm = (to_html(hm_fig, include_plotlyjs=False, full_html=False,
                      div_id=f"hm{n}", config=PLOTLY_CONFIG) if hm_fig else "")
        n += 1

        sections += f"""
        <section>
          <h2>{html.escape(dept)} · under5</h2>
          {best_banner(drows)}
          <h3>Comparación de modelos (menor = mejor; MDA mayor = mejor)</h3>
          {comparison_table(drows)}

          <h3>Serie de tiempo — backtest + proyección futura</h3>
          <p class="cap">Clic en la leyenda para aislar modelos. La línea punteada es el pronóstico de las próximas 4 semanas (más allá del último dato real).</p>
          <div class="chart">{ts}</div>

          <h3>Métrica por horizonte — todos los modelos</h3>
          <p class="cap">Usa los botones para cambiar la métrica (MAE / RMSE / SMAPE / MDA).</p>
          <div class="chart">{mh}</div>

          <h3>Mapa de calor del error (MAE) por año</h3>
          <p class="cap">Rojo = error alto (peor), verde = error bajo (mejor).</p>
          <div class="chart">{hm}</div>

          {commands_block(drows)}
        </section>"""

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neumonía — Dashboard interactivo de modelos</title>
<script type="text/javascript">{get_plotlyjs()}</script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin:0;
         background:#f1f5f9; color:#0f172a; }}
  header {{ background:#0f172a; color:#fff; padding:20px 32px; }}
  header h1 {{ margin:0 0 4px; font-size:20px; }}
  header p {{ margin:0; color:#94a3b8; font-size:13px; }}
  main {{ max-width:1100px; margin:0 auto; padding:22px 18px 60px; }}
  section {{ margin-bottom:44px; }}
  section > h2 {{ font-size:19px; border-left:4px solid #2563eb; padding-left:10px; }}
  section > h3 {{ font-size:14px; color:#334155; margin:20px 0 2px; }}
  .cap {{ font-size:12px; color:#64748b; margin:0 0 8px; }}
  .summary {{ background:#ecfdf5; border:1px solid #a7f3d0; padding:8px 12px;
             border-radius:8px; font-size:14px; }}
  .chart {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px;
           padding:8px 10px; margin:14px 0; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  table.cmp {{ border-collapse:collapse; width:100%; font-size:13px; background:#fff;
             border-radius:8px; overflow:hidden; }}
  table.cmp th, table.cmp td {{ border:1px solid #e2e8f0; padding:5px 8px; text-align:right; }}
  table.cmp th {{ background:#f8fafc; }}
  table.cmp td.mcol {{ text-align:left; font-weight:600; background:#f8fafc; }}
  table.cmp td.best {{ background:#dcfce7; font-weight:700; }}
  details {{ margin-top:12px; font-size:13px; }}
  summary {{ cursor:pointer; color:#2563eb; }}
  .cmdrow {{ margin:8px 0; }}
  .cmdrow code {{ display:block; font-family:Consolas,monospace; font-size:12px;
                background:#0f172a; color:#e2e8f0; padding:7px 10px; border-radius:6px;
                margin-top:3px; white-space:pre-wrap; word-break:break-word; }}
</style></head>
<body>
<header>
  <h1>Neumonía — Dashboard interactivo de modelos (walk-forward + proyección futura)</h1>
  <p>Generado {generated} · Clic en la leyenda para encender/apagar modelos ·
     La línea punteada es el pronóstico de las próximas semanas.</p>
</header>
<main>{sections}</main>
</body></html>"""


def main() -> int:
    rows = collect()
    if not rows:
        print("No hay experimentos. Corre run_grid.py primero.")
        return 1
    out = PRUEBAS_DIR / "dashboard_interactive.html"
    out.write_text(build(rows), encoding="utf-8")
    n_fut = sum(1 for r in rows if r["fut"] is not None)
    print(f"✓ Dashboard interactivo: {out}")
    print(f"  {len(rows)} experimentos · {len({r['department'] for r in rows})} depto(s) · "
          f"{n_fut} con proyección futura")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
