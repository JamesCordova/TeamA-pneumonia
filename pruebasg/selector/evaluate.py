"""
Evaluación estadística de la comparación entre modelos.

El problema que resuelve este módulo: en varios departamentos la diferencia de
MAE entre el mejor modelo y el segundo es de centésimas de caso, muy por debajo
del error del propio estimador. Declarar «ganador» a esa diferencia y construir
sobre ella las reglas del selector sería ajustar reglas al ruido.

Por eso, en lugar de un ganador único, se determina el CONJUNTO EMPATADO: el
mejor modelo más todos aquellos cuya diferencia con él no es estadísticamente
distinguible. El selector no se evalúa por acertar el mínimo exacto, sino por
recomendar un modelo dentro de ese conjunto, que es la decisión que realmente
importa en ingeniería.

La comparación es pareada por origen de pronóstico: ambos modelos se evalúan
sobre exactamente las mismas semanas, de modo que la dificultad intrínseca de
cada semana se cancela en la diferencia. Con horizonte 1 y paso 4, las semanas
evaluadas no se solapan, por lo que las diferencias pueden tratarse como
independientes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Modelos que el selector puede recomendar. SeasonalNaive queda fuera a
# propósito: es el piso de referencia, no una alternativa de ingeniería.
CANDIDATOS = ["SARIMA", "RandomForest", "XGBoost", "LSTM"]
REFERENCIA = "SeasonalNaive"


@dataclass
class ComparacionDepartamento:
    """Resultado de comparar todos los modelos en un departamento."""

    departamento: str
    horizonte: int
    mae: Dict[str, float]
    ee: Dict[str, float]                       # error estándar del MAE
    mejor: str
    conjunto_empatado: List[str]
    p_valores: Dict[str, float]                # p de la comparación contra el mejor
    n_origenes: int
    mejora_sobre_referencia: Dict[str, float] = field(default_factory=dict)

    def regret(self, modelo: str) -> float:
        """Exceso de MAE de `modelo` respecto del mejor, en casos."""
        return self.mae[modelo] - self.mae[self.mejor]

    def regret_relativo(self, modelo: str) -> float:
        """Exceso de MAE como fracción del MAE del mejor."""
        base = self.mae[self.mejor]
        return (self.mae[modelo] - base) / base if base > 0 else np.nan


def cargar_errores(
    grid_dir: Path,
    departamento: str,
    horizonte: int = 1,
    modelos: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Errores absolutos por fecha y modelo para un departamento.

    Returns:
        DataFrame indexado por fecha, una columna por modelo, con |y - ŷ|.
        Solo se conservan las fechas presentes en todos los modelos, que es
        la condición para que la comparación sea pareada.
    """
    modelos = modelos or ([*CANDIDATOS, REFERENCIA])
    series = {}

    # El nombre de la carpeta del snapshot deriva de la etiqueta del
    # experimento, no del modelo, así que el modelo se lee de metrics.json.
    for sub in sorted(grid_dir.iterdir()):
        pred = sub / "predictions.csv"
        met = sub / "metrics.json"
        if not (pred.exists() and met.exists()):
            continue
        info = json.loads(met.read_text(encoding="utf-8-sig"))
        if info.get("department", "").upper() != departamento.upper():
            continue
        modelo = info.get("model")
        if modelo not in modelos:
            continue
        df = pd.read_csv(pred, parse_dates=["date"])
        df = df[df["horizon"] == horizonte]
        s = (df.set_index("date")["actual"] - df.set_index("date")["predicted"]).abs()
        series[modelo] = s[~s.index.duplicated(keep="first")]

    if not series:
        raise FileNotFoundError(f"Sin predicciones para {departamento} en {grid_dir}")
    return pd.DataFrame(series).dropna()


def comparar(
    errores: pd.DataFrame,
    departamento: str,
    horizonte: int = 1,
    alpha: float = 0.05,
) -> ComparacionDepartamento:
    """
    Determina el mejor modelo y el conjunto estadísticamente empatado con él.

    Para cada candidato se contrasta, mediante una prueba t pareada sobre las
    diferencias de error absoluto por origen, la hipótesis de que su error medio
    es igual al del mejor. Los que no se rechazan forman el conjunto empatado.
    """
    candidatos = [c for c in CANDIDATOS if c in errores.columns]
    mae = {m: float(errores[m].mean()) for m in errores.columns}
    ee = {m: float(errores[m].std(ddof=1) / np.sqrt(len(errores))) for m in errores.columns}

    mejor = min(candidatos, key=lambda m: mae[m])

    p_valores: Dict[str, float] = {}
    empatados = [mejor]
    for m in candidatos:
        if m == mejor:
            p_valores[m] = 1.0
            continue
        dif = errores[m] - errores[mejor]
        if np.allclose(dif, 0):
            p = 1.0
        else:
            p = float(stats.ttest_rel(errores[m], errores[mejor]).pvalue)
        p_valores[m] = p
        if p >= alpha:
            empatados.append(m)

    mejora = {}
    if REFERENCIA in mae and mae[REFERENCIA] > 0:
        for m in candidatos:
            mejora[m] = (mae[REFERENCIA] - mae[m]) / mae[REFERENCIA]

    return ComparacionDepartamento(
        departamento=departamento,
        horizonte=horizonte,
        mae=mae,
        ee=ee,
        mejor=mejor,
        conjunto_empatado=sorted(empatados, key=lambda m: mae[m]),
        p_valores=p_valores,
        n_origenes=len(errores),
        mejora_sobre_referencia=mejora,
    )


def comparar_todos(
    grid_dir: Path,
    departamentos: List[str],
    horizonte: int = 1,
    alpha: float = 0.05,
) -> Dict[str, ComparacionDepartamento]:
    """Aplica `comparar` a cada departamento disponible."""
    out: Dict[str, ComparacionDepartamento] = {}
    for dept in departamentos:
        try:
            err = cargar_errores(grid_dir, dept, horizonte=horizonte)
        except FileNotFoundError:
            continue
        out[dept] = comparar(err, dept, horizonte=horizonte, alpha=alpha)
    return out


def tabla_comparaciones(comps: Dict[str, ComparacionDepartamento]) -> pd.DataFrame:
    """Resumen tabular de las comparaciones, una fila por departamento."""
    filas = []
    for dept, c in comps.items():
        fila = {"departamento": dept, "n_origenes": c.n_origenes, "mejor": c.mejor}
        for m in [*CANDIDATOS, REFERENCIA]:
            if m in c.mae:
                fila[f"mae_{m}"] = round(c.mae[m], 3)
        fila["ee_mejor"] = round(c.ee[c.mejor], 3)
        fila["conjunto_empatado"] = " | ".join(c.conjunto_empatado)
        fila["n_empatados"] = len(c.conjunto_empatado)
        filas.append(fila)
    return pd.DataFrame(filas)
