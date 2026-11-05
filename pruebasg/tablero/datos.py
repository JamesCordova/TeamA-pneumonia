"""
Armado del bloque de datos que el tablero incrusta en el HTML.

Todo lo que el navegador necesita se resuelve aquí: el tablero no consulta
ninguna API. La contrapartida es que el archivo lleva los datos dentro, así
que cada estructura se guarda en la forma más compacta que siga siendo legible
(series como listas paralelas al eje de semanas, no como listas de objetos).

Fuentes y qué aporta cada una:

    iras_weekly_clean.csv ............ serie semanal observada, por grupo de edad
    departmental_annual_variation_rates.csv ... población anual, para la tasa
    datos_capitulos.json ............. desempeño, conjuntos empatados y criterio
    grid_selector/*/ ................. predicciones retrospectivas y métricas
    future/futuro_*.csv .............. pronóstico de las 4 semanas siguientes
    profiles/perfiles_*.csv .......... perfil estadístico de cada serie
    shap/ ............................ atribución global y local
    selector/reglas.csv .............. enunciado de las reglas del criterio
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import geometria

MODELOS = ["SARIMA", "RandomForest", "XGBoost", "LSTM", "SeasonalNaive"]
CANDIDATOS = ["SARIMA", "RandomForest", "XGBoost", "LSTM"]
# Familias con pronóstico fuera de muestra. LSTM queda fuera a propósito: el
# criterio la descarta antes de entrenar en los 25 departamentos (regla R0), de
# modo que publicar un pronóstico suyo contradiría el propio criterio.
CON_FUTURO = ["SeasonalNaive", "SARIMA", "RandomForest", "XGBoost"]
COHORTES = [("under5", "Menores de 5"), ("60plus", "60 y más")]
ANIOS_CANAL = 6          # temporadas con canal endémico precalculado
HORIZONTE = 4


def _num(x, nd=3):
    """Float serializable: NaN e infinitos salen como None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _lista(serie, nd=1):
    return [_num(v, nd) for v in serie]


# ---------------------------------------------------------------------------
# Eje temporal y series observadas
# ---------------------------------------------------------------------------

def _eje_y_series(ruta: Path):
    """Rejilla semanal común, semana epidemiológica y series por grupo."""
    df = pd.read_csv(ruta, parse_dates=["week_start"])
    eje = pd.date_range(df["week_start"].min(), df["week_start"].max(), freq="7D")
    iso = eje.isocalendar()
    se = [[int(a), int(s)] for a, s in zip(iso["year"], iso["week"])]

    series = {}
    for grupo, _ in COHORTES:
        sub = df[df["age_group"] == grupo]
        tabla = sub.pivot_table(index="week_start", columns="department",
                                values="cases", aggfunc="sum").reindex(eje)
        series[grupo] = {d: _lista(tabla[d], 1) for d in tabla.columns}
    return eje, se, series


def _poblacion(ruta: Path) -> dict:
    """{departamento: {año: habitantes}} — la base de la tasa por 100 000."""
    r = pd.read_csv(ruta)
    out: dict[str, dict] = {}
    for dept, grupo in r.groupby("department"):
        out[geometria.norm(dept)] = {int(a): int(p) for a, p in
                                     zip(grupo["year"], grupo["population"])
                                     if pd.notna(p)}
    return out


# ---------------------------------------------------------------------------
# Canal endémico
# ---------------------------------------------------------------------------

def _canales(eje, se, series, endemico) -> dict:
    """
    {grupo: {departamento: {año: [[Q1 x53], [mediana x53], [Q3 x53]]}}}

    Se precalculan solo las últimas temporadas, que son las que el tablero
    puede mostrar; el canal de un año depende de las cinco temporadas que lo
    preceden, así que no puede resolverse en el navegador.
    """
    anios = sorted({a for a, _ in se})[-ANIOS_CANAL:]
    salida: dict = {}
    for grupo, _ in COHORTES:
        salida[grupo] = {}
        for dept, valores in series[grupo].items():
            s = pd.Series([np.nan if v is None else v for v in valores], index=eje)
            por_anio = {}
            for anio in anios:
                canal = endemico.canal(s, anio)
                if not canal:
                    continue
                lo, mid, hi = [], [], []
                for semana in range(1, 54):
                    c = canal.get(semana)
                    lo.append(_num(c["lo"], 1) if c else None)
                    mid.append(_num(c["mid"], 1) if c else None)
                    hi.append(_num(c["hi"], 1) if c else None)
                por_anio[str(anio)] = [lo, mid, hi]
            salida[grupo][dept] = por_anio
    return salida


# ---------------------------------------------------------------------------
# Validación retrospectiva
# ---------------------------------------------------------------------------

def _banda_empirica(d: pd.DataFrame, horizonte: int, nivel: float = 0.80):
    """
    Banda empírica de un horizonte, a partir de los errores del walk-forward.

    No es un intervalo de predicción del modelo —los modelos de árboles no lo
    producen— sino la dispersión observada de sus propios errores: los
    cuantiles del error se suman al pronóstico puntual. Es el procedimiento más
    simple que respeta lo que el modelo realmente erró, y el tablero lo declara
    como tal en lugar de presentarlo como un intervalo de confianza.
    """
    sub = d[d["horizon"] == horizonte]
    if sub.empty:
        return None
    err = (sub["actual"] - sub["predicted"]).values
    lo, hi = np.quantile(err, [(1 - nivel) / 2, 1 - (1 - nivel) / 2])
    return [_num(lo, 2), _num(hi, 2)]


def _retrospectiva(grid: Path, eje) -> tuple[dict, dict, dict, dict]:
    """
    Predicciones de horizonte 1, métricas, bandas y cobertura de la banda.

    Las métricas se recalculan desde predictions.csv y no se leen de
    metrics.json. El motivo es la coherencia con el resto de la
    investigación: `selector/evaluate.py` —que es de donde salen el conjunto
    empatado y la evaluación del criterio— puntúa sobre ese mismo archivo, y
    el tablero no puede mostrar una cifra distinta de la que sostiene la
    tabla comparativa. Las dos series difieren (ver
    `pruebasg/chequeo_exclusion_covid.py`).
    """
    pos = {d.date().isoformat(): i for i, d in enumerate(eje)}
    backtest: dict = {}
    metricas: dict = {}
    bandas: dict = {}
    cobertura: dict = {}

    for sub in sorted(grid.iterdir()):
        met, pred = sub / "metrics.json", sub / "predictions.csv"
        if not (met.exists() and pred.exists()):
            continue
        info = json.loads(met.read_text(encoding="utf-8-sig"))
        dept, modelo = geometria.norm(info["department"]), info["model"]
        d = pd.read_csv(pred, parse_dates=["date"])

        h1 = d[d["horizon"] == 1].drop_duplicates("date").sort_values("date")
        idx, val = [], []
        for fecha, p in zip(h1["date"], h1["predicted"]):
            i = pos.get(fecha.date().isoformat())
            if i is not None:
                idx.append(i)
                val.append(_num(p, 1))
        backtest.setdefault(dept, {})[modelo] = {"i": idx, "v": val}

        err = (h1["actual"] - h1["predicted"]).values
        real = h1["actual"].values
        conteo = real != 0
        metricas.setdefault(dept, {})[modelo] = {
            "mae": _num(np.abs(err).mean(), 3),
            "rmse": _num(np.sqrt((err ** 2).mean()), 3),
            "mape": _num(100 * np.abs(err[conteo] / real[conteo]).mean(), 2) if conteo.any() else None,
            "n": int(len(h1)),
        }
        bandas.setdefault(dept, {})[modelo] = [
            _banda_empirica(d, h) for h in range(1, HORIZONTE + 1)]

        # Cobertura de la banda del 80 %, estimada FUERA DE MUESTRA.
        #
        # Medirla sobre los mismos errores con los que se calcularon los
        # cuantiles daría ~80 % por construcción, y no diría nada. Los
        # cuantiles se calculan aquí con el primer 70 % de los orígenes y la
        # cobertura se cuenta sobre el 30 % final, que el cálculo no vio.
        dentro = total = 0
        for h in range(1, HORIZONTE + 1):
            sh = d[d["horizon"] == h].sort_values("date")
            if len(sh) < 20:
                continue
            corte_i = int(len(sh) * 0.70)
            err_cal = (sh["actual"] - sh["predicted"]).values[:corte_i]
            err_ev = (sh["actual"] - sh["predicted"]).values[corte_i:]
            lo, hi = np.quantile(err_cal, [0.10, 0.90])
            dentro += int(((err_ev >= lo) & (err_ev <= hi)).sum())
            total += len(err_ev)
        cobertura.setdefault(dept, {})[modelo] = _num(100 * dentro / total, 1) if total else None

    return backtest, metricas, bandas, cobertura


def _futuro(carpeta: Path, eje) -> tuple[dict, list]:
    """{grupo: {departamento: {modelo: [4 valores]}}} y las 4 fechas."""
    salida: dict = {}
    fechas: list = []
    for grupo, _ in COHORTES:
        ruta = carpeta / f"futuro_{grupo}.csv"
        if not ruta.exists():
            salida[grupo] = {}
            continue
        f = pd.read_csv(ruta)
        salida[grupo] = {}
        for (dept, modelo), g in f.groupby(["departamento", "modelo"]):
            g = g.sort_values("paso")
            salida[grupo].setdefault(geometria.norm(dept), {})[modelo] = \
                _lista(g["prediccion"], 1)
            if not fechas:
                fechas = [str(x) for x in g["fecha"]]
    return salida, fechas


# ---------------------------------------------------------------------------
# Perfil, criterio y explicabilidad
# ---------------------------------------------------------------------------

CAMPOS_PERFIL = ["region", "n_semanas", "media_casos", "mediana_casos", "max_casos",
                 "cv", "indice_dispersion", "prop_ceros", "entropia_espectral",
                 "fuerza_tendencia", "fuerza_estacional", "semana_pico",
                 "acf_1", "acf_4", "acf_52", "estacionaria", "adf_p", "kpss_p",
                 "cat_volumen", "cat_estacionalidad", "cat_entropia",
                 "cat_dispersion", "cat_estacionariedad"]


def _perfiles_y_criterio(carpeta: Path, seleccionar) -> tuple[dict, dict]:
    """Perfil de cada serie y la recomendación que el criterio deriva de él."""
    perfiles: dict = {}
    criterio: dict = {}
    for grupo, _ in COHORTES:
        ruta = carpeta / f"perfiles_{grupo}.csv"
        if not ruta.exists():
            perfiles[grupo], criterio[grupo] = {}, {}
            continue
        p = pd.read_csv(ruta)
        perfiles[grupo], criterio[grupo] = {}, {}
        for _, fila in p.iterrows():
            dept = geometria.norm(fila["departamento"])
            d = {}
            for c in CAMPOS_PERFIL:
                if c not in fila:
                    continue
                v = fila[c]
                d[c] = (bool(v) if isinstance(v, (bool, np.bool_))
                        else v if isinstance(v, str) else _num(v, 4))
            perfiles[grupo][dept] = d

            rec = seleccionar(fila.to_dict())
            criterio[grupo][dept] = {
                "modelo": rec.modelo, "regla": rec.regla,
                "justificacion": rec.justificacion,
                "descartados": dict(rec.descartados),
                "costo_s": _num(rec.costo_estimado_s, 1),
                "costo_todo_s": _num(rec.costo_evaluar_todo_s, 1),
                "ahorro_pct": _num(100 * rec.ahorro_relativo, 1),
            }
    return perfiles, criterio


FAMILIAS_SHAP = ["inercia_corta", "ventanas_moviles", "memoria_media",
                 "calendario_estacional", "rezago_anual", "tendencia"]


def _shap(carpeta: Path, tesis: dict) -> dict:
    """{departamento: {modelo: {familias, top, local}}} y la lista disponible."""
    familias: dict = {}
    for fila in tesis.get("shap_familias", []):
        dept = geometria.norm(fila["departamento"])
        familias.setdefault(dept, {})[fila["modelo"]] = {
            "familias": {f: _num(fila.get(f, 0.0), 4) for f in FAMILIAS_SHAP},
            "top": [], "local": None,
        }

    for ruta in sorted(carpeta.glob("importancia_*.csv")):
        resto = ruta.stem[len("importancia_"):]
        for modelo in ("RandomForest", "XGBoost"):
            if resto.endswith("_" + modelo):
                dept = geometria.norm(resto[: -len(modelo) - 1].replace("_", " "))
                imp = pd.read_csv(ruta)
                col = imp.columns[0]
                nodo = familias.setdefault(dept, {}).setdefault(
                    modelo, {"familias": {}, "top": [], "local": None})
                nodo["top"] = [[str(n), _num(v, 3)] for n, v in
                               zip(imp[col].head(12), imp["shap_medio_abs"].head(12))]
                break

    locales = carpeta / "explicaciones_locales.json"
    if locales.exists():
        for loc in json.loads(locales.read_text(encoding="utf-8")):
            dept = geometria.norm(loc["departamento"])
            nodo = familias.setdefault(dept, {}).setdefault(
                loc["modelo"], {"familias": {}, "top": [], "local": None})
            nodo["local"] = loc
    return familias


# ---------------------------------------------------------------------------
# Ensamblado
# ---------------------------------------------------------------------------

def construir(rutas: dict, endemico, seleccionar) -> dict:
    eje, se, series = _eje_y_series(rutas["semanal"])
    mapa = geometria.construir(rutas["geojson"])
    tesis = json.loads(Path(rutas["tesis"]).read_text(encoding="utf-8"))

    backtest, metricas, bandas, cobertura = _retrospectiva(Path(rutas["grid"]), eje)
    futuro, fechas_futuro = _futuro(Path(rutas["futuro"]), eje)
    perfiles, criterio = _perfiles_y_criterio(Path(rutas["perfiles"]), seleccionar)
    shap = _shap(Path(rutas["shap"]), tesis)
    canal = _canales(eje, se, series, endemico)

    # Desempeño y conjunto empatado, por departamento (los 25)
    evaluacion: dict = {}
    for fila in tesis.get("desempeno", []):
        dept = geometria.norm(fila["departamento"])
        evaluacion[dept] = {
            "region": fila.get("region"),
            "n_origenes": fila.get("n_origenes"),
            "mejor": fila.get("mejor"),
            "empate": [m for m in fila.get("conjunto_empatado", [])],
            "mae": {m: _num(fila.get(m), 3) for m in MODELOS if fila.get(m) is not None},
        }
    for clave in ("criterio_en_muestra", "criterio_fuera_muestra"):
        bloque = tesis.get(clave, {})
        for fila in bloque.get("filas", []):
            dept = geometria.norm(fila["departamento"])
            evaluacion.setdefault(dept, {}).update({
                "regla": fila.get("regla"), "recomendado": fila.get("recomendado"),
                "dentro": fila.get("dentro"), "regret_pct": fila.get("regret_pct"),
                "regret_casos": fila.get("regret_casos"),
                "muestra": clave == "criterio_en_muestra",
            })

    reglas = pd.read_csv(rutas["reglas"]).to_dict("records")

    ultima = len(eje) - 1
    return {
        "meta": {
            "version": rutas.get("version", "2.0"),
            "generado": date.today().isoformat(),
            "ultima_fecha": eje[ultima].date().isoformat(),
            "ultima_se": se[ultima],
            "n_semanas": len(eje),
            "cohortes": [{"id": c, "nombre": n} for c, n in COHORTES],
            "modelos": MODELOS,
            "candidatos": CANDIDATOS,
            "con_futuro": CON_FUTURO,
            "horizonte": HORIZONTE,
            "configuracion": tesis.get("configuracion", {}),
            "muestra": sorted(geometria.norm(f["departamento"])
                              for f in tesis.get("criterio_en_muestra", {}).get("filas", [])),
            "shap_disponible": sorted(shap),
        },
        "mapa": mapa,
        "semanas": [d.date().isoformat() for d in eje],
        "se": se,
        "serie": series,
        "poblacion": _poblacion(rutas["rates"]),
        "canal": canal,
        "perfil": perfiles,
        "criterio": criterio,
        "evaluacion": evaluacion,
        "metricas": metricas,
        "banda": bandas,
        "cobertura": cobertura,
        "backtest": backtest,
        "futuro": futuro,
        "futuro_fechas": fechas_futuro,
        "shap": shap,
        "shap_corr": tesis.get("shap_correlaciones", []),
        "reglas": reglas,
        "estrategias": {
            "en_muestra": tesis.get("criterio_en_muestra", {}),
            "fuera_muestra": tesis.get("criterio_fuera_muestra", {}),
        },
        "resumen": {
            "empates": tesis.get("empates", {}),
            "pertenencia": tesis.get("pertenencia", {}),
            "ganadores": tesis.get("ganadores", {}),
            "mejora": tesis.get("mejora_sobre_referencia", {}),
            "perfil": tesis.get("perfil_resumen", {}),
            "fallos": tesis.get("fallos_fuera_muestra", []),
        },
    }
