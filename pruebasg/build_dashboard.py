#!/usr/bin/env python
"""
Se muestra, agrupado por departamento:
  - Una tabla-resumen y un gráfico de barras comparando el MAE de cada
    experimento (modelo + parámetros).
  - Una tarjeta por experimento con: el COMANDO exacto ejecutado, la tabla de
    métricas por horizonte, y el gráfico de backtest (real vs. predicho).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import sys
from datetime import datetime

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _common import (
    METRIC_HEADERS,
    PRUEBAS_DIR,
    RESULTS_DIR,
    load_json,
    load_manifest,
)

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6",
           "#0891b2", "#db2777", "#65a30d"]


# Helpers de gráficos -> base64

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def backtest_chart(pred_csv, label: str, color: str, year: int | None) -> str | None:
    if pred_csv is None or not pred_csv.exists():
        return None
    df = pd.read_csv(pred_csv, parse_dates=["date"])
    df = df[df["split"] == "backtest"].sort_values("date")
    if df.empty:
        return None
    if year is not None:
        df = df[df["date"].dt.year == year]
        if df.empty:
            return None

    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.plot(df["date"], df["actual"], color="#111827", lw=1.2, label="Real")
    ax.plot(df["date"], df["predicted"], color=color, lw=1.4, alpha=0.9,
            label="Predicho (backtest)")
    ax.set_title(label, fontsize=10)
    ax.set_ylabel("Casos")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.25)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    return _fig_to_base64(fig)


def comparison_chart(rows: list[dict], metric: str = "mae") -> str | None:
    """Barras: métrica (a h=1 y h=max) por experimento del departamento."""
    labels, vals_short, vals_long = [], [], []
    for r in rows:
        by_h = r["by_horizon"]
        if not by_h:
            continue
        hs = sorted(by_h)
        labels.append(r["label"])
        vals_short.append(by_h[hs[0]].get(metric, float("nan")))
        vals_long.append(by_h[hs[-1]].get(metric, float("nan")))
    if not labels:
        return None

    import numpy as np
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.7), 3.6))
    b1 = ax.bar(x - w / 2, vals_short, w, color=PALETTE[0], label="h=1 sem")
    b2 = ax.bar(x + w / 2, vals_long, w, color=PALETTE[1], alpha=0.75, label="h=4 sem")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h == h:  # no NaN
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel(METRIC_HEADERS[metric])
    ax.set_title(f"Comparación de {METRIC_HEADERS[metric]} (menor = mejor)",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_base64(fig)


# Carga de datos desde el manifest

def collect_rows() -> list[dict]:
    rows = []
    for entry in load_manifest():
        if not entry.get("ok"):
            continue
        by_horizon = {}
        config = dict(entry.get("config", {}))
        metrics_file = entry.get("metrics_file")
        if metrics_file:
            mpath = RESULTS_DIR / metrics_file
            if mpath.exists():
                mj = load_json(mpath)
                by_horizon = {int(h): v for h, v in mj.get("metrics_by_horizon", {}).items()}
                config.setdefault("n_steps", mj.get("n_steps"))
        rows.append({
            "label": entry["label"],
            "department": entry["department"],
            "model": entry["model"],
            "command": entry["command"],
            "config": config,
            "timestamp": entry.get("timestamp", ""),
            "pred_csv": (RESULTS_DIR / entry["pred_file"]) if entry.get("pred_file") else None,
            "by_horizon": by_horizon,
        })
    return rows

# Render HTML

def metrics_table_html(by_horizon: dict) -> str:
    if not by_horizon:
        return "<p class='muted'>Sin métricas disponibles.</p>"
    keys = list(METRIC_HEADERS)
    head = "".join(f"<th>{METRIC_HEADERS[k]}</th>" for k in keys)
    body = ""
    # mejor MAE (fila resaltada) por horizonte no aplica; resaltamos min por columna
    hs = sorted(by_horizon)
    best = {k: min((by_horizon[h].get(k, float("inf")) for h in hs)) for k in keys}
    for h in hs:
        cells = ""
        for k in keys:
            v = by_horizon[h].get(k)
            txt = "—" if v is None else f"{v:.2f}"
            cls = " class='best'" if (v is not None and abs(v - best[k]) < 1e-9
                                      and k in ("mae", "rmse", "smape")) else ""
            cells += f"<td{cls}>{txt}</td>"
        body += f"<tr><td class='hcol'>h={h}</td>{cells}</tr>"
    return (f"<table class='metrics'><thead><tr><th>Horizonte</th>{head}</tr>"
            f"</thead><tbody>{body}</tbody></table>")


def experiment_card(row: dict, color: str, year: int | None) -> str:
    img = backtest_chart(row["pred_csv"], row["label"], color, year)
    img_html = (f"<img src='data:image/png;base64,{img}' alt='backtest'/>"
                if img else "<p class='muted'>Sin gráfico de backtest.</p>")
    cfg = row["config"]
    cfg_txt = ""
    if cfg:
        cfg_txt = (f" · window={cfg.get('window_type','?')}"
                   f" · train={cfg.get('train_size','?')}"
                   f" · refit_every={cfg.get('refit_every','?')}"
                   f" · steps={cfg.get('n_steps', '')}")
    return f"""
    <div class="card">
      <div class="card-head">
        <span class="dot" style="background:{color}"></span>
        <h3>{html.escape(row['label'])}</h3>
        <span class="badge">{html.escape(row['model'])}</span>
      </div>
      <div class="cmd">$ {html.escape(row['command'])}</div>
      <div class="cfg muted">{cfg_txt}</div>
      <div class="grid2">
        <div>{metrics_table_html(row['by_horizon'])}</div>
        <div class="chart">{img_html}</div>
      </div>
    </div>"""


def best_summary(rows: list[dict]) -> str:
    scored = []
    for r in rows:
        if not r["by_horizon"]:
            continue
        h1 = min(r["by_horizon"])  # horizonte más corto disponible
        mae = r["by_horizon"][h1].get("mae")
        if mae is not None:
            scored.append((r["label"], mae))
    if not scored:
        return ""
    best = min(scored, key=lambda t: t[1])
    return (f"<p class='summary'>🏆 Mejor a corto plazo (MAE h=1): "
            f"<b>{html.escape(best[0])}</b> ({best[1]:.2f} casos)</p>")


def build_html(rows: list[dict], year: int | None) -> str:
    depts = sorted({r["department"] for r in rows})
    sections = ""
    for dept in depts:
        drows = [r for r in rows if r["department"] == dept]
        colors = {r["label"]: PALETTE[i % len(PALETTE)] for i, r in enumerate(drows)}
        comp = comparison_chart(drows, "mae")
        comp_html = (f"<img src='data:image/png;base64,{comp}' alt='comparación'/>"
                     if comp else "")
        cards = "".join(experiment_card(r, colors[r["label"]], year) for r in drows)
        sections += f"""
        <section>
          <h2>{html.escape(dept)} · under5</h2>
          {best_summary(drows)}
          <div class="comp">{comp_html}</div>
          {cards}
        </section>"""

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    year_note = f" · backtest filtrado al año {year}" if year else ""
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pronóstico de neumonía — Comparación de modelos</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #f1f5f9; color: #0f172a; }}
  header {{ background: #0f172a; color: #fff; padding: 22px 32px; }}
  header h1 {{ margin: 0 0 4px; font-size: 20px; }}
  header p {{ margin: 0; color: #94a3b8; font-size: 13px; }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 24px 20px 60px; }}
  section {{ margin-bottom: 40px; }}
  section > h2 {{ font-size: 18px; border-left: 4px solid #2563eb; padding-left: 10px; }}
  .summary {{ background: #ecfdf5; border: 1px solid #a7f3d0; padding: 8px 12px;
             border-radius: 8px; font-size: 14px; }}
  .comp {{ text-align: center; margin: 12px 0 20px; }}
  .comp img, .chart img {{ max-width: 100%; height: auto; border-radius: 6px; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
          padding: 16px 18px; margin-bottom: 18px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .card-head {{ display: flex; align-items: center; gap: 10px; }}
  .card-head h3 {{ margin: 0; font-size: 15px; flex: 0 0 auto; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
  .badge {{ background: #e0e7ff; color: #3730a3; font-size: 11px; padding: 2px 8px;
           border-radius: 999px; font-weight: 600; }}
  .cmd {{ font-family: Consolas, Menlo, monospace; font-size: 12.5px; background: #0f172a;
         color: #e2e8f0; padding: 9px 12px; border-radius: 8px; margin: 10px 0 4px;
         white-space: pre-wrap; word-break: break-word; line-height: 1.5; }}
  .cfg {{ font-size: 12px; margin-bottom: 10px; }}
  .muted {{ color: #64748b; }}
  .grid2 {{ display: grid; grid-template-columns: minmax(240px, 320px) 1fr; gap: 18px;
           align-items: start; }}
  @media (max-width: 720px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
  table.metrics {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.metrics th, table.metrics td {{ border: 1px solid #e2e8f0; padding: 5px 8px;
         text-align: right; }}
  table.metrics th {{ background: #f8fafc; }}
  table.metrics td.hcol {{ text-align: left; font-weight: 600; background: #f8fafc; }}
  table.metrics td.best {{ background: #dcfce7; font-weight: 700; }}
</style></head>
<body>
<header>
  <h1>Pronóstico de neumonía — Comparación walk-forward de modelos</h1>
  <p>Generado {generated}{year_note} · MAE/RMSE/SMAPE en casos; MDA en % (mayor mejor).
     Celdas verdes = mejor valor por métrica.</p>
</header>
<main>{sections}</main>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=None,
                        help="Filtra el gráfico de backtest a un solo año (ej. 2022)")
    parser.add_argument("--output", default=str(PRUEBAS_DIR / "dashboard.html"),
                        help="Ruta del HTML de salida")
    args = parser.parse_args()

    rows = collect_rows()
    if not rows:
        print("No hay experimentos en pruebasg/results/manifest.json.")
        print("Corre primero:  python pruebasg/run_grid.py")
        return 1

    out = build_html(rows, args.year)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"✓ Dashboard generado: {args.output}")
    print(f"  {len(rows)} experimentos · {len({r['department'] for r in rows})} departamento(s)")
    print("  Ábrelo con doble clic o arrástralo al navegador.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
