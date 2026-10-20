"""
Perfilador de metafeatures de series semanales de notificación obligatoria.

Extrae un vector de descriptores estadísticos de una serie de casos semanales.
Ese vector es la ENTRADA del selector automático de modelos (ver rules.py): la
decisión de qué familia de modelo usar se toma a partir del perfil, nunca a
partir del nombre del departamento ni de la enfermedad, de modo que el criterio
sea transferible a cualquier otra serie de vigilancia epidemiológica.

Las metafeatures siguen la línea de trabajo de meta-aprendizaje para selección
de modelos de series de tiempo (Hyndman et al., FFORMS/FFORMA; paquete
`tsfeatures`), adaptadas aquí a series semanales de vigilancia:
    - se fija el periodo estacional en 52 semanas,
    - se añaden descriptores propios de datos de conteo (sobredispersión,
      proporción de ceros) que no aparecen en el conjunto estándar,
    - se añade la detección de cambios de nivel, frecuente en series de
      notificación por cambios administrativos de cobertura.

Todas las funciones operan sobre `pd.Series` con DatetimeIndex semanal.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import signal, stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

# Periodo estacional de una serie semanal.
PERIODO_ESTACIONAL = 52

# Umbrales de corte para convertir metafeatures numéricas en categorías.
# Se documentan aquí (y no en rules.py) porque describen la SERIE, no la
# decisión: rules.py consume las categorías ya resueltas.
UMBRALES = {
    # Fuerza estacional F_S (Wang, Smith & Hyndman 2006). Por debajo de 0.30 la
    # componente estacional explica menos varianza que el residuo y modelos con
    # estacionalidad explícita dejan de compensar su costo.
    "estacionalidad": {"debil": 0.30, "fuerte": 0.60},
    # Volumen medio de casos semanales. Series de bajo conteo son ruidosas en
    # términos relativos y favorecen modelos rígidos; series de alto volumen
    # toleran modelos flexibles sin sobreajustar.
    "volumen": {"bajo": 10.0, "alto": 60.0},
    # Entropía espectral normalizada: 0 = totalmente predecible (un solo pico
    # espectral), 1 = ruido blanco.
    "entropia": {"baja": 0.55, "alta": 0.75},
    # Índice de dispersión (var/media). = 1 bajo Poisson puro; > 5 indica
    # sobredispersión marcada.
    "dispersion": {"moderada": 5.0},
}


# ---------------------------------------------------------------------------
# Metafeatures individuales
# ---------------------------------------------------------------------------

def _fuerzas_stl(serie: pd.Series, periodo: int = PERIODO_ESTACIONAL) -> Dict[str, float]:
    """
    Fuerza de tendencia y de estacionalidad vía descomposición STL.

        F_T = max(0, 1 - Var(R) / Var(T + R))
        F_S = max(0, 1 - Var(R) / Var(S + R))

    Ambas en [0, 1]: 0 = la componente no aporta nada sobre el residuo,
    cercano a 1 = la componente domina la variabilidad de la serie.
    """
    if len(serie) < 2 * periodo + 1:
        return {"fuerza_tendencia": np.nan, "fuerza_estacional": np.nan,
                "semana_pico": np.nan, "amplitud_estacional": np.nan}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = STL(serie, period=periodo, robust=True).fit()

    resid = res.resid
    var_r = float(np.var(resid, ddof=1))

    var_tr = float(np.var(res.trend + resid, ddof=1))
    var_sr = float(np.var(res.seasonal + resid, ddof=1))

    f_t = max(0.0, 1.0 - var_r / var_tr) if var_tr > 0 else 0.0
    f_s = max(0.0, 1.0 - var_r / var_sr) if var_sr > 0 else 0.0

    # Semana del año en la que la componente estacional alcanza su máximo
    # (promediada sobre todos los años): describe CUÁNDO ocurre el pico.
    est = pd.DataFrame({
        "seasonal": res.seasonal.values,
        "week": serie.index.isocalendar().week.astype(int).values,
    })
    perfil = est.groupby("week")["seasonal"].mean()
    semana_pico = int(perfil.idxmax()) if len(perfil) else np.nan

    # Amplitud estacional relativa al nivel medio de la serie.
    media = float(serie.mean())
    amplitud = float(perfil.max() - perfil.min()) / media if media > 0 else np.nan

    return {
        "fuerza_tendencia": round(float(f_t), 4),
        "fuerza_estacional": round(float(f_s), 4),
        "semana_pico": semana_pico,
        "amplitud_estacional": round(amplitud, 4) if amplitud == amplitud else np.nan,
    }


def _estacionariedad(serie: pd.Series) -> Dict[str, float]:
    """
    Pruebas de raíz unitaria ADF y KPSS.

    Se reportan ambas porque tienen hipótesis nulas opuestas y su combinación
    es más informativa que cualquiera por separado:
        ADF  H0: hay raíz unitaria (no estacionaria)  -> p < 0.05 => estacionaria
        KPSS H0: la serie es estacionaria             -> p < 0.05 => NO estacionaria
    """
    y = serie.dropna().astype(float)
    out: Dict[str, float] = {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["adf_p"] = round(float(adfuller(y, autolag="AIC")[1]), 4)
    except Exception:
        out["adf_p"] = np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["kpss_p"] = round(float(kpss(y, regression="c", nlags="auto")[1]), 4)
    except Exception:
        out["kpss_p"] = np.nan

    adf_ok = out["adf_p"] == out["adf_p"] and out["adf_p"] < 0.05
    kpss_ok = out["kpss_p"] == out["kpss_p"] and out["kpss_p"] >= 0.05
    out["estacionaria"] = bool(adf_ok and kpss_ok)
    # Nº de diferencias regulares sugerido: 0 si ambas pruebas coinciden en
    # estacionariedad, 1 en caso contrario (el desacuerdo se resuelve
    # conservadoramente hacia diferenciar).
    out["d_sugerido"] = 0 if out["estacionaria"] else 1
    return out


def _autocorrelaciones(serie: pd.Series, periodo: int = PERIODO_ESTACIONAL) -> Dict[str, float]:
    """ACF en los rezagos informativos: 1 (inercia), 4 (mes) y 52 (anual)."""
    nlags = max(periodo + 1, 8)
    if len(serie) <= nlags:
        return {"acf_1": np.nan, "acf_4": np.nan, "acf_52": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals = acf(serie.dropna().astype(float), nlags=nlags, fft=True)
    return {
        "acf_1": round(float(vals[1]), 4),
        "acf_4": round(float(vals[4]), 4),
        "acf_52": round(float(vals[periodo]), 4),
    }


def _entropia_espectral(serie: pd.Series) -> float:
    """
    Entropía espectral normalizada en [0, 1]: mide cuánta estructura
    aprovechable tiene la serie.

    Se calcula sobre la densidad espectral (Welch) normalizada a densidad de
    probabilidad. Un espectro concentrado en pocas frecuencias (serie muy
    regular) da entropía baja; ruido blanco da entropía cercana a 1.
    """
    y = serie.dropna().astype(float).values
    if len(y) < 16:
        return float("nan")
    y = y - y.mean()
    if np.allclose(y, 0):
        return float("nan")
    _, psd = signal.welch(y, nperseg=min(len(y), 256))
    psd = psd[psd > 0]
    if psd.size == 0:
        return float("nan")
    p = psd / psd.sum()
    return round(float(-(p * np.log(p)).sum() / np.log(len(p))), 4)


def _estabilidad(serie: pd.Series, periodo: int = PERIODO_ESTACIONAL) -> Dict[str, float]:
    """
    Estabilidad e irregularidad ("lumpiness") sobre bloques anuales disjuntos:
    varianza de las medias y varianza de las varianzas por bloque, normalizadas.

    Detectan si el comportamiento de la serie cambia a lo largo del tiempo, lo
    que penaliza a los modelos entrenados con ventana fija.
    """
    y = serie.dropna().astype(float).values
    n_bloques = len(y) // periodo
    if n_bloques < 2:
        return {"estabilidad": np.nan, "irregularidad": np.nan}
    bloques = y[: n_bloques * periodo].reshape(n_bloques, periodo)
    medias = bloques.mean(axis=1)
    varianzas = bloques.var(axis=1, ddof=1)
    escala = y.mean() ** 2 if y.mean() > 0 else 1.0
    return {
        "estabilidad": round(float(np.var(medias, ddof=1) / escala), 4),
        "irregularidad": round(float(np.var(varianzas, ddof=1) / (escala ** 2)), 6),
    }


def _cambio_nivel(serie: pd.Series, ventana: int = PERIODO_ESTACIONAL) -> float:
    """
    Mayor salto relativo entre medias móviles anuales consecutivas y disjuntas.

    Aproxima una ruptura estructural de nivel (p. ej. un cambio administrativo
    de cobertura de notificación). Se expresa como fracción del nivel medio.
    """
    y = serie.dropna().astype(float)
    if len(y) < 2 * ventana:
        return float("nan")
    medias = y.rolling(ventana).mean().dropna().values[::ventana]
    if len(medias) < 2:
        return float("nan")
    nivel = float(y.mean())
    if nivel <= 0:
        return float("nan")
    return round(float(np.max(np.abs(np.diff(medias))) / nivel), 4)


def _atipicos(serie: pd.Series, periodo: int = PERIODO_ESTACIONAL) -> float:
    """
    Proporción de semanas atípicas: residuos STL fuera de 3·IQR.

    Se mide sobre el residuo (no sobre la serie cruda) para no contar los
    picos estacionales esperados como atípicos.
    """
    if len(serie) < 2 * periodo + 1:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resid = STL(serie, period=periodo, robust=True).fit().resid
    q1, q3 = np.percentile(resid, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return float("nan")
    fuera = (resid < q1 - 3 * iqr) | (resid > q3 + 3 * iqr)
    return round(float(fuera.mean()), 4)


# ---------------------------------------------------------------------------
# Categorización
# ---------------------------------------------------------------------------

def _categorizar(mf: Dict[str, float]) -> Dict[str, str]:
    """
    Traduce las metafeatures numéricas a las categorías que consume el selector.

    Mantener la categorización separada del cálculo permite auditar y ajustar
    los cortes (UMBRALES) sin recalcular los perfiles.
    """
    u = UMBRALES
    fs = mf.get("fuerza_estacional")
    vol = mf.get("media_casos")
    ent = mf.get("entropia_espectral")
    disp = mf.get("indice_dispersion")

    def _bucket(valor, cortes, etiquetas, defecto="desconocido"):
        if valor is None or valor != valor:
            return defecto
        for corte, etiqueta in zip(cortes, etiquetas):
            if valor < corte:
                return etiqueta
        return etiquetas[-1]

    return {
        "cat_estacionalidad": _bucket(
            fs, [u["estacionalidad"]["debil"], u["estacionalidad"]["fuerte"]],
            ["debil", "moderada", "fuerte"],
        ),
        "cat_volumen": _bucket(
            vol, [u["volumen"]["bajo"], u["volumen"]["alto"]],
            ["bajo", "medio", "alto"],
        ),
        "cat_entropia": _bucket(
            ent, [u["entropia"]["baja"], u["entropia"]["alta"]],
            ["baja", "media", "alta"],
        ),
        "cat_dispersion": _bucket(
            disp, [u["dispersion"]["moderada"]],
            ["moderada", "alta"],
        ),
        "cat_estacionariedad": (
            "estacionaria" if mf.get("estacionaria") else "no_estacionaria"
        ),
    }


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def perfilar_serie(
    serie: pd.Series,
    etiqueta: Optional[str] = None,
    periodo: int = PERIODO_ESTACIONAL,
) -> Dict:
    """
    Calcula el perfil completo de metafeatures de una serie semanal.

    Args:
        serie:   casos semanales con DatetimeIndex regular de 7 días.
        etiqueta: nombre para identificar el perfil (p. ej. 'LORETO/under5').
        periodo: periodo estacional en semanas (52 por defecto).

    Returns:
        Diccionario plano con identificación, metafeatures numéricas y
        categorías derivadas. Plano a propósito: se vuelca directo a
        DataFrame/JSON y a la tabla del documento de tesis.
    """
    y = serie.dropna().astype(float)
    if len(y) < periodo:
        raise ValueError(
            f"La serie tiene {len(y)} semanas; se requieren al menos {periodo}."
        )

    media = float(y.mean())
    desv = float(y.std(ddof=1))
    varianza = float(y.var(ddof=1))

    mf: Dict[str, float] = {
        # --- escala y cobertura ---
        "n_semanas": int(len(y)),
        "anio_inicio": int(y.index.min().year),
        "anio_fin": int(y.index.max().year),
        "media_casos": round(media, 3),
        "mediana_casos": round(float(y.median()), 3),
        "max_casos": round(float(y.max()), 3),
        # --- variabilidad ---
        "cv": round(desv / media, 4) if media > 0 else np.nan,
        # Índice de dispersión: =1 bajo Poisson, >1 sobredispersión.
        "indice_dispersion": round(varianza / media, 4) if media > 0 else np.nan,
        "prop_ceros": round(float((y == 0).mean()), 4),
        "asimetria": round(float(stats.skew(y)), 4),
        "curtosis": round(float(stats.kurtosis(y)), 4),
        # --- predictibilidad y estabilidad ---
        "entropia_espectral": _entropia_espectral(y),
        "prop_atipicos": _atipicos(y, periodo),
        "cambio_nivel": _cambio_nivel(y, periodo),
    }
    mf.update(_fuerzas_stl(y, periodo))
    mf.update(_estacionariedad(y))
    mf.update(_autocorrelaciones(y, periodo))
    mf.update(_estabilidad(y, periodo))

    perfil: Dict = {"etiqueta": etiqueta or "serie"}
    perfil.update(mf)
    perfil.update(_categorizar(mf))
    return perfil


def perfilar_departamentos(
    departamentos,
    age_group: str = "under5",
    periodo: int = PERIODO_ESTACIONAL,
) -> pd.DataFrame:
    """
    Perfila varios departamentos y devuelve un DataFrame (una fila por serie).

    Usa `pneumonia.models.utils.get_departmental_data`, es decir exactamente la
    misma carga de datos que usan los modelos: el perfil describe la serie que
    el modelo va a ver, incluida la truncación automática por sub-registro.
    """
    from pneumonia.models.utils import get_departmental_data

    filas = []
    for dept in departamentos:
        serie = get_departmental_data(dept, age_group=age_group)
        perfil = perfilar_serie(serie, etiqueta=f"{dept}/{age_group}", periodo=periodo)
        perfil["departamento"] = dept
        perfil["age_group"] = age_group
        filas.append(perfil)

    df = pd.DataFrame(filas)
    cols = ["departamento", "age_group"] + [
        c for c in df.columns if c not in ("departamento", "age_group", "etiqueta")
    ]
    return df[cols]
