#!/usr/bin/env python
r"""
Tablero v2.0 — Pronóstico de neumonía por departamento (Perú).

Qué es
------
Un prototipo de tablero de vigilancia con la misma estructura de lectura que
PanViz 2.0 —cabecera, cinta de titulares, mapa a la izquierda y paneles a la
derecha— más dos pestañas que no existen en esa plataforma y que son la
aportación de esta tesina:

  * **Criterio de selección** — el perfil estadístico de la serie, la regla
    que se activa, la familia recomendada, el conjunto de modelos
    estadísticamente equivalentes al mejor y el plano volumen x estacionalidad
    donde el criterio traza sus fronteras.
  * **SHAP** — en qué estructura se apoya el modelo recomendado: atribución
    global por familia de variables y descomposición local de una predicción
    concreta.

El resto del armazón es deliberadamente igual al de la plataforma del grupo de
investigación. El tablero no es la contribución: es el vehículo para mostrarla,
y que ambos se lean como una sola plataforma es lo que deja a la vista qué
parte es nueva.

Cómo está hecho
---------------
Un único archivo HTML sin servidor ni dependencias: los datos se incrustan
como JSON y todo el dibujo —mapa incluido— se resuelve a mano sobre SVG. Así
el prototipo se abre desde el disco, se graba y se entrega sin instalar nada.

Las piezas viven en pruebasg/tablero/:
    geometria.py    proyección del GeoJSON a trazos SVG
    datos.py        armado del bloque de datos
    estilos.css     hoja de estilo
    app.js          lógica de la interfaz
    plantilla.html  armazón

Uso:
    python pruebasg/build_frontend_v2.py
    python pruebasg/build_frontend_v2.py --salida otro.html
"""

from __future__ import annotations

import argparse
import json
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

import endemico
from _common import RESULTS_DIR
from selector.rules import seleccionar
from tablero import datos as armador

logging.disable(logging.WARNING)

VERSION = "2.0"
TABLERO = PRUEBAS_DIR / "tablero"
SALIDA = PRUEBAS_DIR / "frontend_neumonia_v2.html"

RUTAS = {
    "semanal": ROOT / "data" / "processed" / "iras_weekly_clean.csv",
    "rates": ROOT / "reports" / "tables" / "departmental_annual_variation_rates.csv",
    "geojson": PRUEBAS_DIR / "peru_departamentos.geojson",
    "tesis": RESULTS_DIR / "tesis" / "datos_capitulos.json",
    "grid": RESULTS_DIR / "grid_selector",
    "futuro": RESULTS_DIR / "future",
    "perfiles": RESULTS_DIR / "profiles",
    "shap": RESULTS_DIR / "shap",
    "reglas": RESULTS_DIR / "selector" / "reglas.csv",
    "version": VERSION,
}


def _faltantes() -> list[str]:
    return [k for k, v in RUTAS.items()
            if isinstance(v, Path) and not Path(v).exists()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--salida", default=str(SALIDA))
    args = p.parse_args()

    faltan = _faltantes()
    if faltan:
        print("Faltan entradas: " + ", ".join(faltan))
        print("Genere primero los resultados correspondientes "
              "(run_grid_selector.py, run_profiles.py, build_future.py, run_shap.py).")
        return 1

    print("Armando el bloque de datos...")
    D = armador.construir(RUTAS, endemico, seleccionar)
    bloque = json.dumps(D, ensure_ascii=False, separators=(",", ":"))

    # El JSON viaja dentro de <script type="application/json">: lo único que hay
    # que impedir es que un "</script>" literal cierre la etiqueta antes de tiempo.
    bloque = bloque.replace("</", "<\\/")

    html = (TABLERO / "plantilla.html").read_text(encoding="utf-8")
    html = html.replace("/*__ESTILOS__*/", (TABLERO / "estilos.css").read_text(encoding="utf-8"))
    html = html.replace("/*__APP__*/", (TABLERO / "app.js").read_text(encoding="utf-8"))
    html = html.replace("/*__VERSION__*/", VERSION)
    html = html.replace("/*__DATOS__*/", bloque)

    destino = Path(args.salida)
    if not destino.is_absolute():
        destino = PRUEBAS_DIR / destino
    destino.write_text(html, encoding="utf-8")

    kb = destino.stat().st_size / 1024
    print(f"\nTablero v{VERSION}: {destino.relative_to(ROOT)}  ({kb:,.0f} KB)".replace(",", " "))
    print(f"  {len(D['mapa']['deptos'])} departamentos · {D['meta']['n_semanas']} semanas · "
          f"última SE {D['meta']['ultima_se'][1]} de {D['meta']['ultima_se'][0]}")
    print(f"  criterio: {len(D['criterio']['under5'])} series perfiladas · "
          f"SHAP: {len(D['meta']['shap_disponible'])} departamentos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
