"""
Selector automático de modelo — versión 1 (reglas sobre el perfil de la serie).

Qué decide y qué no
-------------------
La evidencia experimental de este trabajo (11 departamentos × 5 modelos,
walk-forward de 183-184 orígenes) muestra que, en 10 de los 11 departamentos,
SARIMA, Random Forest y XGBoost producen errores estadísticamente
indistinguibles entre sí. Un selector que pretendiera acertar cuál de los tres
tiene el MAE mínimo estaría ajustándose al ruido.

Por eso el selector no persigue el mínimo exacto. Persigue tres decisiones que
sí son distinguibles y que sí tienen consecuencia de ingeniería:

  1. Qué familias descartar sin evaluarlas. El aprendizaje profundo es
     significativamente peor en 10 de 11 departamentos, con un exceso de error
     medio del 50 %. Descartarlo a priori a partir del perfil ahorra la familia
     más costosa de entrenar sin pérdida de exactitud.
  2. Cuándo el modelo interpretable y barato basta. En los departamentos donde
     SARIMA pertenece al conjunto empatado, recomendarlo no cuesta exactitud
     medible y aporta intervalos de predicción y parámetros interpretables.
  3. Cuándo la flexibilidad se paga. Solo donde el volumen es alto el modelo de
     árboles potenciados supera significativamente al estadístico (Lima
     +17,1 %, Piura +12,6 % frente a SARIMA).

El criterio se expresa sobre el perfil estadístico de la serie (profiler.py) y
no sobre el nombre del departamento ni de la enfermedad, de modo que sea
aplicable a cualquier otra serie de notificación obligatoria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Costo medio de una corrida walk-forward completa, en segundos, medido sobre
# los 55 experimentos de la muestra estratificada. Se usa para cuantificar el
# ahorro de la recomendación frente a evaluar todas las familias.
COSTO_S = {
    "SeasonalNaive": 8.1,
    "SARIMA": 24.0,
    "XGBoost": 36.0,
    "RandomForest": 38.3,
    "LSTM": 81.6,
}

# Parámetros entrenables de la red recurrente en su configuración por defecto
# (48 y 24 unidades, densa de 16, salida de 4). Véase la demostración D.12.
PARAMETROS_LSTM = 17_076


@dataclass
class Recomendacion:
    """Salida del selector para una serie."""

    modelo: str
    regla: str
    justificacion: str
    descartados: Dict[str, str] = field(default_factory=dict)
    costo_estimado_s: float = 0.0
    costo_evaluar_todo_s: float = 0.0

    @property
    def ahorro_relativo(self) -> float:
        if self.costo_evaluar_todo_s <= 0:
            return 0.0
        return 1.0 - self.costo_estimado_s / self.costo_evaluar_todo_s


# ---------------------------------------------------------------------------
# Reglas de descarte
# ---------------------------------------------------------------------------

def _descartar(perfil: Dict, train_size: int, lookback: int = 52) -> Dict[str, str]:
    """
    Familias que el perfil permite excluir antes de entrenarlas.

    El criterio para el aprendizaje profundo es la razón entre el número de
    secuencias de entrenamiento disponibles y el número de parámetros de la
    red. Una ventana de `train_size` semanas con retrospección `lookback`
    produce aproximadamente train_size - lookback secuencias; contrastada con
    los 17 076 parámetros de la arquitectura empleada, la red queda en régimen
    de fuerte sobreparametrización.
    """
    descartados: Dict[str, str] = {}

    n_secuencias = max(0, train_size - lookback)
    if n_secuencias < PARAMETROS_LSTM:
        razon = (
            f"{n_secuencias} secuencias de entrenamiento frente a "
            f"{PARAMETROS_LSTM:,} parámetros de la red "
            f"(razón 1:{PARAMETROS_LSTM / max(n_secuencias, 1):.0f}); "
            "la arquitectura no es identificable con esta longitud de serie"
        ).replace(",", " ")
        descartados["LSTM"] = razon

    return descartados


# ---------------------------------------------------------------------------
# Reglas de selección
# ---------------------------------------------------------------------------

def seleccionar(
    perfil: Dict,
    train_size: int = 520,
    lookback: int = 52,
    candidatos: Optional[List[str]] = None,
) -> Recomendacion:
    """
    Recomienda una familia de modelo a partir del perfil de la serie.

    Args:
        perfil: salida de `profiler.perfilar_serie` (o una fila de la tabla de
            perfiles). Se usan `media_casos`, `fuerza_estacional` y las
            categorías derivadas.
        train_size: ventana de entrenamiento en semanas del esquema de
            validación con el que se usará el modelo.
        lookback: retrospección de la red recurrente, para el criterio de descarte.
        candidatos: familias admisibles; por defecto todas menos la de referencia.

    Returns:
        Recomendacion con el modelo, la regla aplicada, su justificación y las
        familias descartadas con el motivo.
    """
    candidatos = candidatos or ["SARIMA", "RandomForest", "XGBoost", "LSTM"]
    descartados = _descartar(perfil, train_size=train_size, lookback=lookback)
    disponibles = [c for c in candidatos if c not in descartados]

    volumen = float(perfil.get("media_casos", 0.0))
    fs = float(perfil.get("fuerza_estacional", 0.0) or 0.0)

    # --- R1: volumen alto y estacionalidad no dominante -> árboles potenciados
    # Con volumen alto el ruido relativo de Poisson (1/raíz(media)) cae por
    # debajo del 16 %, de modo que la estructura no lineal que el modelo
    # flexible explota es señal y no ruido de notificación.
    if volumen >= 40.0 and fs < 0.60 and "XGBoost" in disponibles:
        return _armar(
            "XGBoost", "R1", disponibles,
            f"Volumen alto ({volumen:.1f} casos/semana, ruido relativo de Poisson "
            f"{100 / volumen ** 0.5:.0f} %) con estacionalidad no dominante "
            f"(F_S = {fs:.2f}). La flexibilidad del modelo de árboles potenciados "
            "es aprovechable: en la muestra supera a SARIMA entre 12,6 % y 17,1 % "
            "en los departamentos que cumplen esta condición.",
            descartados,
        )

    # --- R2: estacionalidad dominante -> modelo estadístico estacional
    # La componente estacional explica la mayor parte de la varianza; el modelo
    # con estacionalidad explícita la representa con 20 parámetros de Fourier
    # y entrega además intervalos de predicción.
    if fs >= 0.60 and "SARIMA" in disponibles:
        return _armar(
            "SARIMA", "R2", disponibles,
            f"Estacionalidad dominante (F_S = {fs:.2f}): la componente estacional "
            "explica la mayor parte de la varianza y la representación de Fourier "
            "la captura de forma parsimoniosa. En la muestra, el modelo flexible "
            "no aporta mejora significativa bajo esta condición y en Puno resulta "
            "3,4 % peor.",
            descartados,
        )

    # --- R3: volumen bajo -> modelo estadístico estacional
    # Con menos de diez casos semanales el ruido relativo supera el 32 %: los
    # rezagos que alimentan a los modelos tabulares son mayoritariamente ruido.
    if volumen < 10.0 and "SARIMA" in disponibles:
        return _armar(
            "SARIMA", "R3", disponibles,
            f"Volumen bajo ({volumen:.1f} casos/semana, ruido relativo de Poisson "
            f"{100 / max(volumen, 1e-9) ** 0.5:.0f} %). Las variables de rezago "
            "están dominadas por la variabilidad de notificación y los modelos "
            "tabulares sobreajustan; el modelo estadístico es preferible y en "
            "Madre de Dios resulta el de menor error.",
            descartados,
        )

    # --- R4: caso restante -> bosque aleatorio
    # Volumen intermedio y estacionalidad moderada: el bosque aleatorio es la
    # opción más estable del conjunto empatado, con menor varianza que los
    # árboles potenciados ante hiperparámetros no ajustados.
    if "RandomForest" in disponibles:
        return _armar(
            "RandomForest", "R4", disponibles,
            f"Volumen intermedio ({volumen:.1f} casos/semana) y estacionalidad "
            f"moderada (F_S = {fs:.2f}). Ninguna condición anterior se cumple; el "
            "bosque aleatorio es la alternativa más robusta del conjunto empatado, "
            "por ser la menos sensible al ajuste de hiperparámetros.",
            descartados,
        )

    # --- Salvaguarda: si todo lo anterior fue descartado
    restante = disponibles[0] if disponibles else "SeasonalNaive"
    return _armar(restante, "R5", disponibles,
                  "Ninguna regla aplicable; se recomienda la única familia disponible.",
                  descartados)


def _armar(modelo: str, regla: str, disponibles: List[str],
           justificacion: str, descartados: Dict[str, str]) -> Recomendacion:
    return Recomendacion(
        modelo=modelo,
        regla=regla,
        justificacion=justificacion,
        descartados=descartados,
        costo_estimado_s=COSTO_S.get(modelo, 0.0),
        costo_evaluar_todo_s=sum(
            COSTO_S.get(m, 0.0) for m in ["SARIMA", "RandomForest", "XGBoost", "LSTM"]
        ),
    )


# Descripción legible de las reglas, para la documentación del criterio.
DESCRIPCION_REGLAS = [
    ("R0", "Descarte de aprendizaje profundo",
     "secuencias de entrenamiento < parámetros de la red",
     "Excluir LSTM",
     "Sobreparametrización verificable antes de entrenar (D.12)."),
    ("R1", "Volumen alto",
     "media >= 40 casos/semana y F_S < 0,60",
     "XGBoost",
     "Ruido relativo de Poisson < 16 %: la estructura no lineal es explotable."),
    ("R2", "Estacionalidad dominante",
     "F_S >= 0,60",
     "SARIMA con Fourier",
     "La estacionalidad explica la mayor parte de la varianza; 20 parámetros bastan."),
    ("R3", "Volumen bajo",
     "media < 10 casos/semana",
     "SARIMA con Fourier",
     "Ruido relativo > 32 %: los rezagos no son informativos."),
    ("R4", "Caso restante",
     "volumen intermedio y estacionalidad moderada",
     "Random Forest",
     "Opción más estable del conjunto empatado."),
]
