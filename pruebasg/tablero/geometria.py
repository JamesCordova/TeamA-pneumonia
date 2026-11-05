"""
Geometría del mapa: el GeoJSON de departamentos se convierte en trazos SVG.

Se resuelve en tiempo de construcción y no en el navegador por dos razones: el
resultado pesa bastante menos que el GeoJSON completo, y el dibujo deja de
depender de una biblioteca de mapas, lo que mantiene el prototipo en un único
archivo que se abre sin servidor.
"""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

ANCHO = 1000.0        # lienzo de referencia; la altura sale de la proyección
TOLERANCIA = 0.45     # simplificación, en unidades del lienzo


def norm(s) -> str:
    """Clave de departamento sin tildes y en mayúsculas."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c)).upper().strip()


def titulo(s: str) -> str:
    """MADRE DE DIOS -> Madre de Dios."""
    menores = {"de", "del", "la", "las", "los", "y"}
    partes = [p.lower() for p in str(s).split()]
    return " ".join(p if i and p in menores else p.capitalize()
                    for i, p in enumerate(partes))


def _mercator(lon: float, lat: float) -> tuple[float, float]:
    lat = max(min(lat, 84.0), -84.0)
    return lon, math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def _simplificar(puntos: list, tol: float) -> list:
    """Douglas-Peucker: descarta los vértices que no cambian la silueta."""
    if len(puntos) < 3:
        return puntos
    ax, ay = puntos[0]
    bx, by = puntos[-1]
    dx, dy = bx - ax, by - ay
    norma = math.hypot(dx, dy)
    peor, idx = -1.0, 0
    for i in range(1, len(puntos) - 1):
        px, py = puntos[i]
        d = (math.hypot(px - ax, py - ay) if norma == 0
             else abs(dy * px - dx * py + bx * ay - by * ax) / norma)
        if d > peor:
            peor, idx = d, i
    if peor <= tol:
        return [puntos[0], puntos[-1]]
    return _simplificar(puntos[:idx + 1], tol)[:-1] + _simplificar(puntos[idx:], tol)


def _dentro(px: float, py: float, anillos: list) -> bool:
    """Punto en polígono con huecos, por la regla par-impar."""
    dentro = False
    for anillo in anillos:
        for i in range(len(anillo) - 1):
            x1, y1 = anillo[i]
            x2, y2 = anillo[i + 1]
            if (y1 > py) != (y2 > py):
                if px < x1 + (py - y1) * (x2 - x1) / (y2 - y1):
                    dentro = not dentro
    return dentro


def _dist_borde(px: float, py: float, anillos: list) -> float:
    mejor = float("inf")
    for anillo in anillos:
        for i in range(len(anillo) - 1):
            x1, y1 = anillo[i]
            x2, y2 = anillo[i + 1]
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
            mejor = min(mejor, math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))
    return mejor


def _punto_interior(anillos: list) -> tuple[float, float]:
    """
    Polo de inaccesibilidad aproximado: el punto interior más alejado del borde.

    El centroide no sirve: en departamentos alargados o cóncavos cae fuera del
    propio polígono y la etiqueta terminaría sobre el mar o sobre el vecino.
    """
    ext = anillos[0]
    xs = [p[0] for p in ext]
    ys = [p[1] for p in ext]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    mejor, mejor_d = ((x0 + x1) / 2, (y0 + y1) / 2), -1.0
    n = 26
    for i in range(1, n):
        for j in range(1, n):
            px = x0 + (x1 - x0) * i / n
            py = y0 + (y1 - y0) * j / n
            if not _dentro(px, py, anillos):
                continue
            d = _dist_borde(px, py, anillos)
            if d > mejor_d:
                mejor, mejor_d = (px, py), d
    return mejor


def construir(ruta_geojson: Path) -> dict:
    """{'ancho', 'alto', 'deptos': [{'id','nombre','d','ex','ey'}]}"""
    geo = json.loads(Path(ruta_geojson).read_text(encoding="utf-8"))
    crudo = []
    for f in geo["features"]:
        g = f["geometry"]
        poligonos = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        crudo.append((norm(f["properties"]["NOMBDEP"]),
                      [[[_mercator(x, y) for x, y in anillo] for anillo in pol]
                       for pol in poligonos]))

    todos = [p for _, pols in crudo for pol in pols for anillo in pol for p in anillo]
    minx, maxx = min(p[0] for p in todos), max(p[0] for p in todos)
    miny, maxy = min(p[1] for p in todos), max(p[1] for p in todos)
    escala = ANCHO / (maxx - minx)

    def a_lienzo(p):
        return (round((p[0] - minx) * escala, 1), round((maxy - p[1]) * escala, 1))

    deptos = []
    for nombre, poligonos in crudo:
        trazos, area_mayor, mayor = [], -1.0, None
        for pol in poligonos:
            anillos = []
            for anillo in pol:
                pts = [a_lienzo(p) for p in anillo]
                limpio = [pts[0]]
                for p in pts[1:]:
                    if p != limpio[-1]:
                        limpio.append(p)
                if limpio[0] != limpio[-1]:
                    limpio.append(limpio[0])
                limpio = _simplificar(limpio, TOLERANCIA)
                if len(limpio) > 3:
                    anillos.append(limpio)
            if not anillos:
                continue
            xs = [p[0] for p in anillos[0]]
            ys = [p[1] for p in anillos[0]]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area > area_mayor:
                area_mayor, mayor = area, anillos
            for anillo in anillos:
                trazos.append("M" + "L".join(f"{x},{y}" for x, y in anillo) + "Z")
        ex, ey = _punto_interior(mayor)
        deptos.append({"id": nombre, "nombre": titulo(nombre), "d": "".join(trazos),
                       "ex": round(ex, 1), "ey": round(ey, 1)})

    deptos.sort(key=lambda d: d["id"])
    return {"ancho": round(ANCHO, 1), "alto": round((maxy - miny) * escala, 1),
            "deptos": deptos}
