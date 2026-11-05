#!/usr/bin/env python
r"""
Canal endémico: qué es una semana "habitual" en un departamento y en qué zona
cae la semana observada.

El canal endémico es el instrumento clásico de la vigilancia epidemiológica:
para cada semana del año se toman las observaciones de esa misma semana en
varias temporadas de referencia y se resumen en tres cortes, que delimitan
cuatro zonas de alerta.

    Q1  -> techo de la zona de éxito
    Q2  -> techo de la zona de seguridad   (la mediana: lo "habitual")
    Q3  -> techo de la zona de alarma      (umbral epidémico)
    > Q3 -> zona epidémica

Dos decisiones explícitas:

* **Temporadas de referencia.** Las cinco anteriores al año consultado. Se
  excluyen 2020 y 2021: el descenso de notificación durante la pandemia no
  describe un comportamiento habitual y arrastraría los cortes hacia abajo
  durante los cinco años siguientes. Es la misma exclusión que aplica la
  validación walk-forward de esta investigación.
* **Método.** Cuartiles sobre los conteos, sin transformación logarítmica.
  La alternativa clásica —media ± 2 desviaciones sobre el logaritmo— supone
  simetría que estas series no tienen; con cinco observaciones por semana el
  cuartil es además menos sensible a un solo año atípico.

El canal NO forma parte del criterio de selección de modelo: es contexto de
lectura para el pronóstico, y se declara como tal.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TEMPORADAS_REF = 5
ANIOS_EXCLUIDOS = (2020, 2021)
ZONAS = ("exito", "seguridad", "alarma", "epidemica")
ETIQUETA_ZONA = {"exito": "ÉXITO", "seguridad": "SEGURIDAD",
                 "alarma": "ALARMA", "epidemica": "EPIDEMIA"}


def anios_referencia(anio: int, inicio: int = 2000,
                     n: int = TEMPORADAS_REF) -> List[int]:
    """Las `n` temporadas anteriores a `anio`, de la más reciente hacia atrás."""
    out: List[int] = []
    y = anio - 1
    while y >= inicio and len(out) < n:
        if y not in ANIOS_EXCLUIDOS:
            out.append(y)
        y -= 1
    return out


def canal(serie: pd.Series, anio: int, inicio: int = 2000) -> Dict[int, dict]:
    """
    Canal endémico de `anio`: un corte por semana epidemiológica.

    Args:
        serie: casos semanales indexados por fecha de inicio de semana.
        anio: año para el que se construye el canal.

    Returns:
        {semana_epidemiologica: {"lo", "mid", "hi", "n"}}, con `n` el número de
        temporadas de referencia que aportaron esa semana.
    """
    refs = anios_referencia(anio, inicio)
    if not refs:
        return {}
    iso = serie.index.isocalendar()
    marco = pd.DataFrame({"valor": serie.values,
                          "anio": iso["year"].values,
                          "semana": iso["week"].values})
    marco = marco[marco["anio"].isin(refs)]
    salida: Dict[int, dict] = {}
    for semana, grupo in marco.groupby("semana"):
        v = grupo["valor"].dropna().values
        if not len(v):
            continue
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        salida[int(semana)] = {"lo": round(float(q1), 2),
                               "mid": round(float(q2), 2),
                               "hi": round(float(q3), 2),
                               "n": int(len(v))}
    return salida


def zona(valor: Optional[float], corte: Optional[dict]) -> Optional[str]:
    """Zona en la que cae `valor` dado el corte de su semana."""
    if valor is None or corte is None or not np.isfinite(valor):
        return None
    if valor <= corte["lo"]:
        return "exito"
    if valor <= corte["mid"]:
        return "seguridad"
    if valor <= corte["hi"]:
        return "alarma"
    return "epidemica"


def exceso_vs_habitual(valor: Optional[float], corte: Optional[dict]) -> Optional[float]:
    """Exceso porcentual de `valor` sobre la mediana de referencia."""
    if valor is None or corte is None or not corte.get("mid"):
        return None
    return round(100.0 * (valor - corte["mid"]) / corte["mid"], 1)
