#!/usr/bin/env python
"""
Demostraciones implementadas de las formulaciones del Capítulo III.

Cada demostración aplica la fórmula del capítulo a los datos reales del estudio
y CONTRASTA el resultado obtenido a mano contra el que produce la
implementación. No es una derivación algebraica: es una verificación numérica
reproducible, con su resultado.

Salidas:
    pruebasg/results/demostraciones/resultados.json
    pruebasg/results/demostraciones/figuras/*.png

Uso:
    python pruebasg/run_demostraciones.py
    python pruebasg/run_demostraciones.py --solo D4 D11
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "demostraciones"
FIG_DIR = OUT_DIR / "figuras"

DEPT = "AMAZONAS"      # departamento de referencia de las demostraciones
AGE = "under5"
TRAIN = 520            # ventana de entrenamiento del esquema de validación

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")


def _serie(dept: str = DEPT, n: int | None = None) -> pd.Series:
    from pneumonia.models.utils import get_departmental_data
    s = get_departmental_data(dept, age_group=AGE)
    return s.iloc[-n:] if n else s


def _r(x, k=6):
    """Redondeo seguro a JSON."""
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), k)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


# ===========================================================================
# D1 — Interpolación lineal de la población
# ===========================================================================

def d1_interpolacion() -> dict:
    """
    Reproduce con la fórmula del capítulo el valor que la base de datos
    almacena para un año intercensal, partiendo de los dos años censales que
    lo delimitan.
    """
    from pneumonia.data.load_data import get_db_engine
    from sqlalchemy import text

    eng = get_db_engine(None)
    with eng.connect() as c:
        fila = pd.read_sql(
            text('SELECT * FROM population_annual_dept WHERE department = :d'),
            c, params={"d": DEPT},
        ).iloc[0]

    a, b, t = 2007, 2017, 2012          # censos que delimitan y año objetivo
    P_a, P_b = float(fila[str(a)]), float(fila[str(b)])
    P_t_formula = P_a + (P_b - P_a) * (t - a) / (b - a)
    P_t_base = float(fila[str(t)])

    # Serie completa del tramo, para la tabla del documento
    tramo = {str(y): int(fila[str(y)]) for y in range(a, b + 1)}
    incrementos = [tramo[str(y + 1)] - tramo[str(y)] for y in range(a, b)]

    return {
        "id": "D1",
        "titulo": "Interpolación lineal de la población",
        "departamento": DEPT,
        "anio_censal_a": a, "P_a": P_a,
        "anio_censal_b": b, "P_b": P_b,
        "anio_objetivo": t,
        "pendiente": _r((P_b - P_a) / (b - a), 4),
        "P_t_por_formula": _r(P_t_formula, 4),
        "P_t_en_base_de_datos": P_t_base,
        "diferencia_absoluta": _r(abs(P_t_formula - P_t_base), 6),
        "coincide": bool(abs(P_t_formula - P_t_base) < 1.0),
        "incrementos_anuales_del_tramo": incrementos,
        # El incremento alterna entre 4137 y 4138 porque la base almacena la
        # población como entero: la pendiente real es 4137,2 y el redondeo se
        # reparte. La constancia del incremento salvo redondeo es la evidencia
        # de que el tramo intercensal es efectivamente lineal.
        "incremento_constante_salvo_redondeo": bool(
            max(incrementos) - min(incrementos) <= 1),
        "tramo": tramo,
    }


# ===========================================================================
# D2 — Tasa de incidencia y su precisión
# ===========================================================================

def d2_tasa_incidencia() -> dict:
    from pneumonia.data.load_data import get_db_engine
    from sqlalchemy import text

    anio = 2019
    s = _serie()
    casos = int(s[s.index.year == anio].sum())

    eng = get_db_engine(None)
    with eng.connect() as c:
        fila = pd.read_sql(
            text('SELECT * FROM population_annual_dept WHERE department = :d'),
            c, params={"d": DEPT},
        ).iloc[0]
    P = float(fila[str(anio)])

    T = casos / P * 1e5
    ee = 1e5 * np.sqrt(casos) / P
    return {
        "id": "D2",
        "titulo": "Tasa de incidencia por cien mil habitantes",
        "departamento": DEPT, "anio": anio,
        "casos_notificados": casos,
        "poblacion": P,
        "proporcion": _r(casos / P, 8),
        "tasa_por_100k": _r(T, 4),
        "error_estandar": _r(ee, 4),
        "ic95_inferior": _r(T - 1.96 * ee, 4),
        "ic95_superior": _r(T + 1.96 * ee, 4),
        "error_relativo_pct": _r(100 * ee / T, 4),
    }


# ===========================================================================
# D3 — Descomposición STL: verificación de la identidad
# ===========================================================================

def d3_descomposicion() -> dict:
    from statsmodels.tsa.seasonal import STL

    s = _serie()
    res = STL(s, period=52, robust=True).fit()
    recompuesta = res.trend + res.seasonal + res.resid
    err = (s - recompuesta).abs()

    var_r = float(np.var(res.resid, ddof=1))
    var_sr = float(np.var(res.seasonal + res.resid, ddof=1))
    var_tr = float(np.var(res.trend + res.resid, ddof=1))
    f_s = max(0.0, 1 - var_r / var_sr)
    f_t = max(0.0, 1 - var_r / var_tr)

    tabla = pd.DataFrame({
        "Y_t": s, "T_t": res.trend, "S_t": res.seasonal, "R_t": res.resid,
        "T+S+R": recompuesta,
    }).head(6).round(4)

    # Figura
    fig, axes = plt.subplots(4, 1, figsize=(9, 6.4), sharex=True)
    for ax, (serie, nombre) in zip(axes, [(s, "Observado $Y_t$"),
                                          (res.trend, "Tendencia $T_t$"),
                                          (res.seasonal, "Estacional $S_t$"),
                                          (res.resid, "Residuo $R_t$")]):
        ax.plot(serie.index, serie.values, lw=0.8, color="#2e5c8a")
        ax.set_ylabel(nombre, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_title(f"Descomposición STL — {DEPT} ({AGE}), período 52 semanas", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "d3_stl.png", dpi=140)
    plt.close(fig)

    return {
        "id": "D3",
        "titulo": "Descomposición STL y verificación de la identidad aditiva",
        "departamento": DEPT,
        "n_semanas": int(len(s)),
        "error_max_identidad": _r(float(err.max()), 10),
        "error_medio_identidad": _r(float(err.mean()), 12),
        "var_residuo": _r(var_r, 4),
        "var_estacional_mas_residuo": _r(var_sr, 4),
        "var_tendencia_mas_residuo": _r(var_tr, 4),
        "fuerza_estacional": _r(f_s, 4),
        "fuerza_tendencia": _r(f_t, 4),
        "tabla_primeras_semanas": tabla.reset_index().rename(
            columns={"index": "semana", "week_start": "semana"}).astype(str).values.tolist(),
        "tabla_columnas": ["semana", "Y_t", "T_t", "S_t", "R_t", "T+S+R"],
        "figura": "d3_stl.png",
    }


# ===========================================================================
# D4 — Dickey-Fuller aumentada: regresión reconstruida a mano
# ===========================================================================

def d4_adf() -> dict:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss

    y = _serie().astype(float)
    # Se usa regression='c' (constante sin tendencia determinista), la misma
    # especificación que emplea el perfilador, para que la conclusión de esta
    # demostración y la de la tabla de perfiles no se contradigan.
    res = adfuller(y, autolag="AIC", regression="c")
    stat, pval, usedlag, nobs, crit, _ = res
    kpss_stat, kpss_p, _, kpss_crit = kpss(y, regression="c", nlags="auto")

    # Reconstrucción manual de la regresión de la ecuación del capítulo:
    #   dy_t = alpha + gamma*y_{t-1} + sum(phi_i * dy_{t-i}) + e_t
    p = usedlag
    dy = y.diff()
    datos = pd.DataFrame({"dy": dy, "y_lag": y.shift(1)})
    for i in range(1, p + 1):
        datos[f"dy_lag{i}"] = dy.shift(i)
    datos = datos.dropna()

    X = sm.add_constant(datos[["y_lag"] + [f"dy_lag{i}" for i in range(1, p + 1)]])
    ols = sm.OLS(datos["dy"], X).fit()
    gamma = float(ols.params["y_lag"])
    t_gamma = float(ols.tvalues["y_lag"])

    return {
        "id": "D4",
        "titulo": "Prueba de Dickey-Fuller aumentada",
        "departamento": DEPT,
        "n_observaciones": int(nobs),
        "rezagos_usados_p": int(p),
        "gamma_estimado": _r(gamma, 6),
        "rho_implicito": _r(1 + gamma, 6),
        "t_gamma_reconstruido": _r(t_gamma, 6),
        "estadistico_adf_libreria": _r(stat, 6),
        "diferencia": _r(abs(t_gamma - stat), 9),
        "coincide": bool(abs(t_gamma - stat) < 1e-6),
        "p_valor": _r(pval, 6),
        "valores_criticos": {k: _r(v, 4) for k, v in crit.items()},
        "decision_adf": ("se rechaza la raíz unitaria" if pval < 0.05
                         else "no se rechaza la raíz unitaria"),
        "kpss_estadistico": _r(float(kpss_stat), 6),
        "kpss_p_valor": _r(float(kpss_p), 6),
        "kpss_critico_5pct": _r(float(kpss_crit["5%"]), 4),
        "decision_kpss": ("se rechaza la estacionariedad" if kpss_p < 0.05
                          else "no se rechaza la estacionariedad"),
        "conclusion_conjunta": (
            "estacionaria" if (pval < 0.05 and kpss_p >= 0.05) else
            "no estacionaria: las dos pruebas no coinciden en estacionariedad, "
            "y el desacuerdo se resuelve conservadoramente hacia diferenciar"
        ),
        "d_sugerido": 0 if (pval < 0.05 and kpss_p >= 0.05) else 1,
    }


# ===========================================================================
# D5 — ACF calculada a mano frente a la librería
# ===========================================================================

def d5_acf() -> dict:
    from statsmodels.tsa.stattools import acf

    y = _serie().astype(float).values
    n = len(y)
    ybar = y.mean()
    den = float(((y - ybar) ** 2).sum())

    manual = {}
    for k in (1, 4, 52):
        num = float(((y[k:] - ybar) * (y[:-k] - ybar)).sum())
        manual[k] = num / den

    lib = acf(y, nlags=52, fft=True)
    banda = 1.96 / np.sqrt(n)

    # Figura del correlograma
    vals = acf(y, nlags=60, fft=True)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(range(len(vals)), vals, width=0.7, color="#2e5c8a")
    ax.axhline(banda, color="#c0504d", ls="--", lw=1)
    ax.axhline(-banda, color="#c0504d", ls="--", lw=1)
    ax.axhline(0, color="#555", lw=0.8)
    ax.annotate(f"±1,96/√n = ±{banda:.4f}", xy=(45, banda), fontsize=9,
                color="#c0504d", va="bottom")
    ax.set_xlabel("Rezago k (semanas)")
    ax.set_ylabel(r"$\hat{\rho}_k$")
    ax.set_title(f"Función de autocorrelación — {DEPT} ({AGE})", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d5_acf.png", dpi=140)
    plt.close(fig)

    return {
        "id": "D5",
        "titulo": "Función de autocorrelación",
        "departamento": DEPT,
        "n": n,
        "media": _r(float(ybar), 4),
        "denominador_gamma0_x_n": _r(den, 2),
        "rezagos": [
            {"k": k,
             "rho_manual": _r(manual[k], 6),
             "rho_libreria": _r(float(lib[k]), 6),
             "diferencia": _r(abs(manual[k] - lib[k]), 9),
             "supera_banda": bool(abs(manual[k]) > banda)}
            for k in (1, 4, 52)
        ],
        "banda_bartlett": _r(float(banda), 6),
        "veces_banda_rezago_52": _r(abs(manual[52]) / banda, 2),
        "figura": "d5_acf.png",
    }


# ===========================================================================
# D6 — PACF: fórmula cerrada frente a Yule-Walker de la librería
# ===========================================================================

def d6_pacf() -> dict:
    from statsmodels.tsa.stattools import acf, pacf

    y = _serie().astype(float).values
    r = acf(y, nlags=3, fft=True)
    phi11 = r[1]
    phi22 = (r[2] - r[1] ** 2) / (1 - r[1] ** 2)

    lib = pacf(y, nlags=3, method="ywm")
    return {
        "id": "D6",
        "titulo": "Función de autocorrelación parcial",
        "departamento": DEPT,
        "rho_1": _r(float(r[1]), 6),
        "rho_2": _r(float(r[2]), 6),
        "phi_11_formula": _r(float(phi11), 6),
        "phi_11_libreria": _r(float(lib[1]), 6),
        "phi_22_formula": _r(float(phi22), 6),
        "phi_22_libreria": _r(float(lib[2]), 6),
        "diferencia_phi22": _r(abs(phi22 - lib[2]), 6),
        "rho1_al_cuadrado": _r(float(r[1] ** 2), 6),
        "comentario_ar1": ("si el proceso fuera AR(1) se cumpliría rho_2 = rho_1^2 "
                           "y phi_22 sería cero"),
    }


# ===========================================================================
# D7 — Ljung-Box calculado a mano frente a la librería
# ===========================================================================

def d7_ljung_box() -> dict:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf
    from scipy import stats as st

    y = _serie().astype(float).values
    n = len(y)
    h = 52
    r = acf(y, nlags=h, fft=True)

    terminos = [(k, float(r[k]), float(r[k] ** 2 / (n - k))) for k in range(1, h + 1)]
    Q_manual = n * (n + 2) * sum(t[2] for t in terminos)
    p_manual = float(st.chi2.sf(Q_manual, df=h))

    lb = acorr_ljungbox(y, lags=[h], return_df=True)
    Q_lib = float(lb["lb_stat"].iloc[0])
    p_lib = float(lb["lb_pvalue"].iloc[0])

    return {
        "id": "D7",
        "titulo": "Estadístico de Ljung-Box",
        "departamento": DEPT,
        "n": n, "h": h,
        "Q_manual": _r(Q_manual, 4),
        "Q_libreria": _r(Q_lib, 4),
        "diferencia": _r(abs(Q_manual - Q_lib), 8),
        "coincide": bool(abs(Q_manual - Q_lib) < 1e-6),
        "p_manual": _r(p_manual, 10),
        "p_libreria": _r(p_lib, 10),
        "valor_critico_chi2_95": _r(float(st.chi2.ppf(0.95, h)), 4),
        "primeros_terminos": [
            {"k": k, "rho_k": _r(rk, 6), "rho2_sobre_n_menos_k": _r(term, 9)}
            for k, rk, term in terminos[:5]
        ],
        "decision": ("se rechaza la independencia: la serie tiene estructura temporal modelable"
                     if p_manual < 0.05 else "no se rechaza la independencia"),
    }


# ===========================================================================
# D8 — SARIMA: costo del período estacional largo, medido
# ===========================================================================

def d8_sarima() -> dict:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = _serie(n=TRAIN).astype(float)
    y.index = pd.DatetimeIndex(y.index).to_period("W")

    # (a) Estacionalidad clásica con s = 52
    t0 = time.time()
    clasico = {"converge": False}
    try:
        mod = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52),
                      enforce_stationarity=False, enforce_invertibility=False)
        fit = mod.fit(disp=False, maxiter=50)
        clasico = {
            "converge": bool(fit.mle_retvals.get("converged", False)),
            "dim_estado": int(mod.k_states),
            "n_parametros": int(len(fit.params)),
            "aic": _r(float(fit.aic), 3),
            "segundos": _r(time.time() - t0, 2),
        }
    except Exception as exc:  # noqa: BLE001
        clasico = {"converge": False, "error": str(exc)[:160],
                   "segundos": _r(time.time() - t0, 2)}

    # (b) Estacionalidad por términos de Fourier como exógenas
    K = 10
    t = np.arange(len(y))
    exog = np.column_stack([f(2 * np.pi * (k + 1) * t / 52)
                            for k in range(K) for f in (np.sin, np.cos)])
    t0 = time.time()
    mod_f = SARIMAX(y, order=(1, 1, 1), exog=exog,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit_f = mod_f.fit(disp=False, maxiter=200)
    fourier = {
        "converge": bool(fit_f.mle_retvals.get("converged", False)),
        "dim_estado": int(mod_f.k_states),
        "n_parametros": int(len(fit_f.params)),
        "aic": _r(float(fit_f.aic), 3),
        "segundos": _r(time.time() - t0, 2),
    }

    return {
        "id": "D8",
        "titulo": "Costo del período estacional largo frente a los términos de Fourier",
        "departamento": DEPT,
        "n_entrenamiento": int(len(y)),
        "clasico_s52": clasico,
        "fourier_K10": fourier,
        "razon_dim_estado": (_r(clasico.get("dim_estado", np.nan) /
                                max(fourier["dim_estado"], 1), 2)
                             if clasico.get("dim_estado") else None),
        "razon_tiempo": (_r(clasico["segundos"] / max(fourier["segundos"], 1e-9), 1)
                         if clasico.get("segundos") else None),
    }


# ===========================================================================
# D9 — Términos de Fourier: ortogonalidad verificada
# ===========================================================================

def d9_fourier() -> dict:
    S, K = 52, 10
    t = np.arange(1, S + 1)
    cols, nombres = [], []
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * t / S)); nombres.append(f"sin_{k}")
        cols.append(np.cos(2 * np.pi * k * t / S)); nombres.append(f"cos_{k}")
    X = np.column_stack(cols)

    G = X.T @ X
    fuera = G - np.diag(np.diag(G))
    max_fuera = float(np.abs(fuera).max())
    diag = np.diag(G)

    return {
        "id": "D9",
        "titulo": "Ortogonalidad de los términos de Fourier",
        "S": S, "K": K,
        "n_parametros_fourier": 2 * K,
        "n_parametros_indicadoras": S - 1,
        "reduccion_pct": _r(100 * (1 - 2 * K / (S - 1)), 2),
        "max_producto_cruzado_fuera_diagonal": _r(max_fuera, 10),
        "ortogonal": bool(max_fuera < 1e-9),
        "valores_diagonal_distintos": sorted({round(float(v), 6) for v in diag}),
        "K_maximo_identificable": S // 2,
    }


# ===========================================================================
# D10 — Random Forest: verificación empírica de la fórmula de varianza
# ===========================================================================

def d10_random_forest() -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from pneumonia.features.build import build_features

    s = _serie()
    X, y = build_features(s, lags=[1, 2, 4, 8, 26, 52], windows=[4, 13, 26, 52])
    corte = len(X) - 104
    Xtr, ytr, Xte = X.iloc[:corte], y.iloc[:corte], X.iloc[corte:]

    # Las cantidades sigma^2 y rho de la fórmula están definidas PARA UN PUNTO
    # FIJO x sobre la aleatoriedad del entrenamiento: sigma^2 es la varianza de
    # la predicción de un árbol en x, y rho la correlación entre las
    # predicciones de dos árboles distintos en ese mismo x. Ninguna de las dos
    # puede estimarse dentro de un único bosque —allí dos árboles dan solo dos
    # números en x—, así que se entrenan R bosques independientes y ambas se
    # estiman a través de las réplicas.
    # Las réplicas deben diferir TAMBIÉN en el conjunto de entrenamiento: si se
    # fija el conjunto y solo se cambia la semilla, la correlación que induce el
    # dato compartido se convierte en un sesgo constante —no en varianza— y rho
    # se estima en cero. Se usan por tanto R ventanas de entrenamiento distintas
    # de la misma serie, todas anteriores al tramo de prueba común.
    R, B_max, VENT = 20, 50, 300
    inicios = np.linspace(0, corte - VENT - 1, R).astype(int)
    bosques = []
    for r, ini in enumerate(inicios):
        Xw = Xtr.iloc[ini:ini + VENT]
        yw = ytr.iloc[ini:ini + VENT]
        f = RandomForestRegressor(n_estimators=B_max, max_depth=10, min_samples_leaf=2,
                                  max_features="sqrt", random_state=7000 + r, n_jobs=-1)
        f.fit(Xw.values, yw.values)
        bosques.append(np.vstack([a.predict(Xte.values) for a in f.estimators_]))
    # Tensor (R x B_max x n_test)
    T = np.stack(bosques)

    # sigma^2: varianza entre réplicas de la predicción del primer árbol,
    # promediada sobre los puntos de prueba.
    sigma2 = float(np.mean(T[:, 0, :].var(axis=0, ddof=1)))

    # rho: correlación entre réplicas de las predicciones de dos árboles
    # distintos en el mismo punto, promediada sobre los puntos de prueba.
    a1, a2 = T[:, 0, :], T[:, 1, :]
    a1c, a2c = a1 - a1.mean(axis=0), a2 - a2.mean(axis=0)
    num = (a1c * a2c).sum(axis=0)
    den = np.sqrt((a1c ** 2).sum(axis=0) * (a2c ** 2).sum(axis=0))
    ok = den > 0
    rho = float(np.mean(num[ok] / den[ok]))

    curva = []
    for Bp in (1, 2, 5, 10, 25, 50):
        # Predicción de un bosque de Bp árboles en cada réplica
        M = T[:, :Bp, :].mean(axis=1)                         # (R x n_test)
        var_emp = float(np.mean(M.var(axis=0, ddof=1)))
        var_teo = rho * sigma2 + (1 - rho) * sigma2 / Bp
        curva.append({"B": Bp, "R_replicas": R,
                      "var_empirica": _r(var_emp, 5),
                      "var_teorica": _r(var_teo, 5),
                      "error_rel_pct": _r(100 * abs(var_emp - var_teo) / var_teo, 2)})
    B_ref = B_max

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    Bs = [c["B"] for c in curva]
    ax.plot(Bs, [c["var_empirica"] for c in curva], "o-", label="Varianza empírica", color="#2e5c8a")
    ax.plot(Bs, [c["var_teorica"] for c in curva], "s--",
            label=r"$\rho\sigma^2 + (1-\rho)\sigma^2/B$", color="#c0504d")
    ax.axhline(rho * sigma2, color="#888", ls=":", label=r"Piso  $\rho\sigma^2$")
    ax.set_xscale("log"); ax.set_xlabel("Número de árboles B")
    ax.set_ylabel("Varianza de la predicción del ensamble")
    ax.set_title(f"Reducción de varianza por agregación — {DEPT}", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG_DIR / "d10_varianza_rf.png", dpi=140)
    plt.close(fig)

    return {
        "id": "D10",
        "titulo": "Reducción de varianza por agregación en Random Forest",
        "departamento": DEPT,
        "B_referencia": B_ref,
        "n_test": int(len(Xte)),
        "sigma2_entre_arboles": _r(sigma2, 5),
        "rho_medio_entre_arboles": _r(rho, 5),
        "piso_rho_sigma2": _r(rho * sigma2, 5),
        "curva": curva,
        "error_rel_medio_pct": _r(float(np.mean([c["error_rel_pct"] for c in curva])), 2),
        "figura": "d10_varianza_rf.png",
    }


# ===========================================================================
# D11 — XGBoost: peso óptimo de hoja reproducido a mano
# ===========================================================================

def d11_xgboost() -> dict:
    import xgboost as xgb
    from xgboost import XGBRegressor
    from pneumonia.features.build import build_features

    s = _serie(n=TRAIN)
    X, y = build_features(s, lags=[1, 2, 4, 8, 13], windows=[4, 13, 26])

    lam, base = 1.0, 0.5
    m = XGBRegressor(n_estimators=1, max_depth=2, learning_rate=1.0, reg_lambda=lam,
                     base_score=base, objective="reg:squarederror", subsample=1.0,
                     colsample_bytree=1.0, min_child_weight=0, gamma=0,
                     tree_method="exact")
    m.fit(X.values, y.values)

    booster = m.get_booster()
    dump = booster.get_dump(dump_format="text")[0]
    hojas = booster.predict(xgb.DMatrix(X.values), pred_leaf=True).astype(int).ravel()

    # Valores de hoja que reporta la librería
    valores_lib = {}
    for linea in dump.splitlines():
        if "leaf=" in linea:
            idx = int(linea.strip().split(":")[0])
            valores_lib[idx] = float(linea.split("leaf=")[1].strip())

    # Gradiente y hessiano de la pérdida cuadrática tal como los define XGBoost:
    #   l = 1/2 (y - yhat)^2  ->  g = yhat - y,  h = 1
    yhat0 = np.full(len(y), base)
    g = yhat0 - y.values
    h = np.ones(len(y))

    filas = []
    for idx in sorted(valores_lib):
        sel = hojas == idx
        G, H = float(g[sel].sum()), float(h[sel].sum())
        w = -G / (H + lam)
        filas.append({
            "hoja": idx, "n_observaciones": int(sel.sum()),
            "G_j": _r(G, 4), "H_j": _r(H, 1),
            "w_formula": _r(w, 6),
            "w_libreria": _r(valores_lib[idx], 6),
            "diferencia": _r(abs(w - valores_lib[idx]), 8),
        })

    return {
        "id": "D11",
        "titulo": "Peso óptimo de hoja en XGBoost",
        "departamento": DEPT,
        "lambda": lam, "base_score": base,
        "n_entrenamiento": int(len(X)),
        "arbol_volcado": dump.strip(),
        "hojas": filas,
        "max_diferencia": _r(max(f["diferencia"] for f in filas), 8),
        "coincide": bool(max(f["diferencia"] for f in filas) < 1e-5),
    }


# ===========================================================================
# D12 — LSTM: conteo de parámetros frente a Keras
# ===========================================================================

def d12_lstm() -> dict:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential

    lookback, d, u1, u2 = 52, 1, 48, 24
    modelo = Sequential([
        Input(shape=(lookback, d)),
        LSTM(u1, return_sequences=True), Dropout(0.3),
        LSTM(u2), Dropout(0.3),
        Dense(16, activation="relu"), Dense(4),
    ])

    def celda(u, ent):
        return 4 * (u * (u + ent) + u)

    capas = [
        {"capa": "LSTM 1", "formula": "4[u(u+d)+u], u=48, d=1", "teorico": celda(u1, d)},
        {"capa": "LSTM 2", "formula": "4[u(u+d)+u], u=24, d=48", "teorico": celda(u2, u1)},
        {"capa": "Dense 16", "formula": "24x16+16", "teorico": u2 * 16 + 16},
        {"capa": "Dense 4", "formula": "16x4+4", "teorico": 16 * 4 + 4},
    ]
    reales = [l.count_params() for l in modelo.layers if l.count_params() > 0]
    for c, r in zip(capas, reales):
        c["keras"] = int(r)
        c["coincide"] = bool(c["teorico"] == r)

    total_teorico = sum(c["teorico"] for c in capas)
    total_keras = int(modelo.count_params())

    n_secuencias = TRAIN - lookback
    decaimiento = [{"norma_W": w, "factor_52_pasos": _r(float(w ** 52), 8)}
                   for w in (0.9, 0.95, 0.99)]

    return {
        "id": "D12",
        "titulo": "Dimensionamiento de la red recurrente",
        "capas": capas,
        "total_teorico": int(total_teorico),
        "total_keras": total_keras,
        "coincide": bool(total_teorico == total_keras),
        "ventana_entrenamiento": TRAIN,
        "lookback": lookback,
        "n_secuencias_entrenamiento": int(n_secuencias),
        "razon_parametros_por_secuencia": _r(total_keras / n_secuencias, 1),
        "decaimiento_gradiente": decaimiento,
    }


# ===========================================================================
# D13 — Métricas: desigualdades verificadas sobre los resultados reales
# ===========================================================================

def d13_metricas() -> dict:
    grid = RESULTS_DIR / "grid_selector"
    perfiles = pd.read_csv(RESULTS_DIR / "profiles" / f"perfiles_{AGE}.csv").set_index("departamento")

    filas, violaciones = [], 0
    for sub in sorted(grid.iterdir()):
        pred, met = sub / "predictions.csv", sub / "metrics.json"
        if not (pred.exists() and met.exists()):
            continue
        info = json.loads(met.read_text(encoding="utf-8-sig"))
        if info.get("model") != "RandomForest":
            continue
        df = pd.read_csv(pred)
        df = df[df["horizon"] == 1]
        e = (df["actual"] - df["predicted"]).values
        n = len(e)
        mae = float(np.abs(e).mean())
        rmse = float(np.sqrt((e ** 2).mean()))
        ceros = int((df["actual"] == 0).sum())
        con_valor = df[df["actual"] != 0]
        mape = float((np.abs(con_valor["actual"] - con_valor["predicted"]) /
                      con_valor["actual"]).mean() * 100) if len(con_valor) else np.nan
        ok = (mae <= rmse + 1e-9) and (rmse <= np.sqrt(n) * mae + 1e-9)
        if not ok:
            violaciones += 1
        dept = info["department"]
        filas.append({
            "departamento": dept,
            "media_casos": _r(float(perfiles.loc[dept, "media_casos"]), 1),
            "n": n, "mae": _r(mae, 3), "rmse": _r(rmse, 3),
            "razon_rmse_mae": _r(rmse / mae, 3),
            "cota_superior_raiz_n": _r(float(np.sqrt(n)), 2),
            "mape_pct": _r(mape, 2),
            "semanas_con_cero": ceros,
            "pct_semanas_sin_mape": _r(100 * ceros / n, 2),
            "cumple_desigualdad": bool(ok),
        })

    filas.sort(key=lambda f: -f["razon_rmse_mae"])
    return {
        "id": "D13",
        "titulo": "Relaciones entre las métricas de error",
        "modelo_usado": "RandomForest",
        "horizonte": 1,
        "n_departamentos": len(filas),
        "violaciones_de_la_desigualdad": violaciones,
        "filas": filas,
        "razon_min": _r(min(f["razon_rmse_mae"] for f in filas), 3),
        "razon_max": _r(max(f["razon_rmse_mae"] for f in filas), 3),
        "mae_min": _r(min(f["mae"] for f in filas), 3),
        "mae_max": _r(max(f["mae"] for f in filas), 3),
    }


# ===========================================================================
# D14 — Número de orígenes del walk-forward
# ===========================================================================

def d14_walkforward() -> dict:
    grid = RESULTS_DIR / "grid_selector"
    from pneumonia.models.utils import get_departmental_data

    filas = []
    vistos = set()
    for sub in sorted(grid.iterdir()):
        met = sub / "metrics.json"
        if not met.exists():
            continue
        info = json.loads(met.read_text(encoding="utf-8-sig"))
        dept = info["department"]
        if dept in vistos:
            continue
        vistos.add(dept)
        cfg = info.get("config", {})
        m, step = int(cfg.get("train_size", TRAIN)), int(cfg.get("step", 4))
        n = len(get_departmental_data(dept, age_group=AGE))
        K_formula = (n - m - 1) // step + 1
        K_real = int(info.get("n_steps", 0))
        filas.append({
            "departamento": dept, "n": n, "m": m, "s": step,
            "K_formula": int(K_formula), "K_reportado": K_real,
            "coincide": bool(K_formula == K_real),
        })
    filas.sort(key=lambda f: f["departamento"])
    return {
        "id": "D14",
        "titulo": "Número de orígenes de la validación de origen móvil",
        "filas": filas,
        "todas_coinciden": bool(all(f["coincide"] for f in filas)),
        "n_departamentos": len(filas),
        "reduccion_error_estandar": _r(float(np.sqrt(np.mean([f["K_real"] if False else f["K_formula"] for f in filas]))), 2),
    }



# ===========================================================================
# D15 — Codificación circular del calendario
# ===========================================================================

def d15_codificacion_circular() -> dict:
    """
    Muestra por qué la semana del año no puede entrar al modelo como número
    ordinal: las semanas 52 y 1 son consecutivas en el tiempo pero distan 51
    unidades en esa escala. La representación en seno y coseno restituye la
    proximidad real.
    """
    S = 52
    ref = 1
    sr, cr = np.sin(2 * np.pi * ref / S), np.cos(2 * np.pi * ref / S)
    filas = []
    for w in (50, 51, 52, 1, 2, 13, 26):
        s, c = np.sin(2 * np.pi * w / S), np.cos(2 * np.pi * w / S)
        filas.append({
            "semana": w,
            "sin": _r(float(s), 4),
            "cos": _r(float(c), 4),
            "distancia_circular_a_semana_1": _r(float(np.hypot(s - sr, c - cr)), 4),
            "distancia_ordinal_a_semana_1": int(abs(w - ref)),
        })
    d52 = next(f for f in filas if f["semana"] == 52)
    d2 = next(f for f in filas if f["semana"] == 2)
    d26 = next(f for f in filas if f["semana"] == 26)
    return {
        "id": "D15",
        "titulo": "Codificación circular de la semana del año",
        "periodo": S,
        "filas": filas,
        "distancia_52_a_1_circular": d52["distancia_circular_a_semana_1"],
        "distancia_52_a_1_ordinal": d52["distancia_ordinal_a_semana_1"],
        "distancia_2_a_1_circular": d2["distancia_circular_a_semana_1"],
        "coinciden_52_y_2": bool(abs(d52["distancia_circular_a_semana_1"] -
                                     d2["distancia_circular_a_semana_1"]) < 1e-9),
        "distancia_maxima_26": d26["distancia_circular_a_semana_1"],
    }


# ===========================================================================
# D16 — Ausencia de fuga de información
# ===========================================================================

def d16_fuga_informacion() -> dict:
    """
    Cuantifica el efecto de construir los estadísticos móviles sin desplazar la
    serie un periodo. Sin el desplazamiento, la media móvil de la ventana que
    termina en t incluye y_t, es decir el valor que se quiere predecir, y el
    error medido cae de forma artificial. Esa caída es la magnitud del
    autoengaño que la convención del capítulo evita.
    """
    from sklearn.ensemble import RandomForestRegressor
    from pneumonia.config import WEEKS_PER_YEAR

    serie = _serie()
    lags, ventanas = [1, 2, 4, 8, 13], [4, 13, 26]

    def construir(con_desplazamiento: bool):
        df = pd.DataFrame(index=serie.index)
        for k in lags:
            df[f"lag_{k}"] = serie.shift(k)
        base = serie.shift(1) if con_desplazamiento else serie
        for w in ventanas:
            df[f"rolling_mean_{w}"] = base.rolling(w).mean()
            df[f"rolling_std_{w}"] = base.rolling(w).std()
        woy = serie.index.isocalendar().week.astype(int)
        df["sin_week"] = np.sin(2 * np.pi * woy / WEEKS_PER_YEAR)
        df["cos_week"] = np.cos(2 * np.pi * woy / WEEKS_PER_YEAR)
        valido = df.notna().all(axis=1)
        return df[valido], serie[valido]

    resultados = {}
    for etiqueta, desplazar in (("correcto", True), ("con_fuga", False)):
        X, y = construir(desplazar)
        corte = int(len(X) * 0.8)
        modelo = RandomForestRegressor(n_estimators=200, max_depth=10,
                                       min_samples_leaf=2, max_features="sqrt",
                                       random_state=42, n_jobs=-1)
        modelo.fit(X.iloc[:corte].values, y.iloc[:corte].values)
        pred = modelo.predict(X.iloc[corte:].values)
        real = y.iloc[corte:].values
        imp = dict(zip(X.columns, modelo.feature_importances_))
        top = max(imp.items(), key=lambda kv: kv[1])
        resultados[etiqueta] = {
            "mae": _r(float(np.abs(real - pred).mean()), 4),
            "rmse": _r(float(np.sqrt(((real - pred) ** 2).mean())), 4),
            "n_entrenamiento": int(corte),
            "n_prueba": int(len(real)),
            "variable_mas_importante": top[0],
            "importancia": _r(float(top[1]), 4),
        }

    mae_ok = resultados["correcto"]["mae"]
    mae_fuga = resultados["con_fuga"]["mae"]
    return {
        "id": "D16",
        "titulo": "Efecto de la fuga de información en los estadísticos móviles",
        "departamento": DEPT,
        "lags": lags, "ventanas": ventanas,
        "correcto": resultados["correcto"],
        "con_fuga": resultados["con_fuga"],
        "reduccion_artificial_pct": _r(100 * (mae_ok - mae_fuga) / mae_ok, 2),
        "veces_menor": _r(mae_ok / mae_fuga, 2) if mae_fuga else None,
    }

DEMOSTRACIONES = {
    "D1": d1_interpolacion, "D2": d2_tasa_incidencia, "D3": d3_descomposicion,
    "D4": d4_adf, "D5": d5_acf, "D6": d6_pacf, "D7": d7_ljung_box,
    "D8": d8_sarima, "D9": d9_fourier, "D10": d10_random_forest,
    "D11": d11_xgboost, "D12": d12_lstm, "D13": d13_metricas,
    "D14": d14_walkforward, "D15": d15_codificacion_circular,
    "D16": d16_fuga_informacion,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solo", nargs="+", default=None,
                        help="Ejecutar solo estas demostraciones (p. ej. D4 D11)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    seleccion = args.solo or list(DEMOSTRACIONES)
    ruta = OUT_DIR / "resultados.json"
    resultados = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}

    for clave in seleccion:
        if clave not in DEMOSTRACIONES:
            print(f"  ! {clave} no existe"); continue
        print(f"  {clave} ...", end="", flush=True)
        t0 = time.time()
        try:
            resultados[clave] = DEMOSTRACIONES[clave]()
            print(f" ok ({time.time() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f" FALLO: {type(exc).__name__}: {str(exc)[:130]}")
            resultados[clave] = {"id": clave, "error": str(exc)[:400]}
        ruta.write_text(json.dumps(resultados, indent=1, ensure_ascii=False,
                                   default=str), encoding="utf-8")

    print(f"\nGuardado: {ruta.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
