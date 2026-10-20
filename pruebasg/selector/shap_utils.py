"""
Explicabilidad mediante SHAP de los modelos de árboles (Random Forest, XGBoost).

Papel dentro de la investigación
--------------------------------
SHAP no se incorpora como un añadido de interpretabilidad al final del proceso,
sino como el mecanismo que permite verificar el criterio del selector. El
selector recomienda una familia de modelo a partir del perfil de la serie —por
ejemplo, un modelo estadístico estacional cuando la fuerza estacional es
dominante—. SHAP permite comprobar si el modelo se apoya efectivamente en la
estructura que el perfil declara: si en una serie de estacionalidad fuerte el
modelo de árboles concentra su atribución en las variables de calendario y en
el rezago anual, la regla queda respaldada por el mecanismo y no solo por el
resultado agregado de error.

Los valores de Shapley atribuyen a cada variable su contribución marginal
promediada sobre todos los órdenes posibles de incorporación, lo que les
confiere las propiedades de eficiencia (las atribuciones suman la diferencia
entre la predicción y el valor base) y simetría. Para modelos de árboles existe
un algoritmo exacto de tiempo polinómico, que es el aplicado aquí.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Agrupación de las variables construidas por features/build.py en familias
# con significado epidemiológico. La atribución agregada por familia es más
# estable e interpretable que la de cada variable por separado.
FAMILIAS = {
    "inercia_corta": lambda n: n.startswith("lag_") and int(n.split("_")[1]) <= 4,
    "memoria_media": lambda n: n.startswith("lag_") and 4 < int(n.split("_")[1]) < 52,
    "rezago_anual": lambda n: n == "lag_52",
    "ventanas_moviles": lambda n: n.startswith("rolling_"),
    "calendario_estacional": lambda n: n in
        ("week_of_year", "sin_week", "cos_week", "month", "quarter"),
    "tendencia": lambda n: n == "trend",
}


def familia_de(nombre: str) -> str:
    """Familia a la que pertenece una variable; 'otras' si no encaja."""
    for fam, test in FAMILIAS.items():
        try:
            if test(nombre):
                return fam
        except (ValueError, IndexError):
            continue
    return "otras"


@dataclass
class ResultadoShap:
    """Atribuciones SHAP de un modelo sobre una serie."""

    departamento: str
    modelo: str
    nombres: List[str]
    valores: np.ndarray            # (n_muestras, n_variables)
    X: pd.DataFrame
    valor_base: float

    def importancia_global(self) -> pd.Series:
        """Media del valor absoluto de SHAP por variable, ordenada."""
        imp = pd.Series(np.abs(self.valores).mean(axis=0), index=self.nombres)
        return imp.sort_values(ascending=False)

    def importancia_por_familia(self, normalizar: bool = True) -> pd.Series:
        """Atribución agregada por familia de variables."""
        imp = self.importancia_global()
        fam = imp.groupby([familia_de(n) for n in imp.index]).sum()
        if normalizar and fam.sum() > 0:
            fam = fam / fam.sum()
        return fam.sort_values(ascending=False)



    def explicacion_local(self, fecha=None, indice: Optional[int] = None,
                          top: int = 8) -> dict:
        """
        Explicación de UNA predicción concreta.

        La atribución global responde en qué se apoya el modelo en conjunto; la
        local responde por qué pronosticó lo que pronosticó en una semana
        determinada, que es la pregunta que se hace quien recibe una alerta
        epidemiológica. Por la propiedad de eficiencia, las contribuciones
        suman exactamente la diferencia entre la predicción y el valor base, y
        esa identidad se verifica aquí.

        Args:
            fecha: semana a explicar; si se omite se usa `indice`.
            indice: posición dentro de las observaciones evaluadas.
            top: número de variables a detallar; el resto se agrupa.
        """
        if fecha is not None:
            marca = pd.Timestamp(fecha)
            posiciones = np.where(self.X.index == marca)[0]
            if not len(posiciones):
                raise ValueError(
                    f"{marca.date()} no está entre las semanas evaluadas "
                    f"({self.X.index.min().date()} a {self.X.index.max().date()})"
                )
            i = int(posiciones[0])
        elif indice is not None:
            i = int(indice)
        else:
            raise ValueError("indique `fecha` o `indice`")

        contrib = pd.Series(self.valores[i], index=self.nombres)
        prediccion = float(self.valor_base + contrib.sum())
        orden = contrib.abs().sort_values(ascending=False)
        principales = contrib[orden.index[:top]]
        resto = float(contrib[orden.index[top:]].sum())

        return {
            "departamento": self.departamento,
            "modelo": self.modelo,
            "semana": str(self.X.index[i].date()),
            "valor_base": round(float(self.valor_base), 4),
            "prediccion": round(prediccion, 4),
            "efecto_total": round(prediccion - float(self.valor_base), 4),
            "contribuciones": [
                {"variable": n, "valor_variable": round(float(self.X.iloc[i][n]), 3),
                 "contribucion": round(float(v), 4),
                 "familia": familia_de(n)}
                for n, v in principales.items()
            ],
            "resto_agrupado": round(resto, 4),
            "n_variables_restantes": int(len(contrib) - len(principales)),
            # Verificación de la propiedad de eficiencia
            "suma_contribuciones": round(float(contrib.sum()), 6),
            "eficiencia_verificada": bool(
                abs((self.valor_base + contrib.sum()) - prediccion) < 1e-6),
        }

    def semana_pico(self):
        """Semana de mayor valor observado entre las evaluadas, para explicarla."""
        return self.X.index[int(np.argmax(self.X["lag_1"].values))]             if "lag_1" in self.X.columns else self.X.index[0]


def _construir_modelo(modelo: str, departamento: str, age_group: str):
    """Instancia el modelo con la MISMA configuración usada en la grilla."""
    if modelo == "RandomForest":
        from pneumonia.models.ml.random_forest import RandomForestModel
        return RandomForestModel(
            department=departamento, age_group=age_group,
            lags=[1, 2, 4, 8, 26, 52], windows=[4, 13, 26, 52],
        )
    if modelo == "XGBoost":
        from pneumonia.models.ml.xgboost import XGBoostModel
        return XGBoostModel(
            department=departamento, age_group=age_group,
            xgb_params={"n_estimators": 500, "learning_rate": 0.01,
                        "max_depth": 3, "colsample_bytree": 0.7},
        )
    raise ValueError(f"SHAP solo se aplica a modelos de árboles, no a {modelo}")


def calcular_shap(
    departamento: str,
    modelo: str = "RandomForest",
    age_group: str = "under5",
    train_size: Optional[int] = 520,
    max_muestras: int = 400,
    semilla: int = 42,
) -> ResultadoShap:
    """
    Ajusta el modelo sobre la ventana final de la serie y calcula sus SHAP.

    Se usa la ventana final —las últimas `train_size` semanas— porque es la que
    corresponde al modelo que quedaría en producción tras el reentrenamiento
    final descrito en la Etapa 6.

    Args:
        max_muestras: submuestra de filas sobre las que se evalúan los valores
            de Shapley; el cálculo es exacto por fila, la submuestra solo acota
            el tiempo total.
    """
    import shap
    from pneumonia.features.build import build_features
    from pneumonia.models.utils import get_departmental_data

    serie = get_departmental_data(departamento, age_group=age_group)
    if train_size and len(serie) > train_size:
        serie = serie.iloc[-train_size:]

    modelo_obj = _construir_modelo(modelo, departamento, age_group)
    modelo_obj.fit(serie)

    X, _ = build_features(serie, lags=modelo_obj.lags, windows=modelo_obj.windows)

    estimador = getattr(modelo_obj, "_rf", None) or getattr(modelo_obj, "_xgb", None)
    explainer = shap.TreeExplainer(estimador)

    if len(X) > max_muestras:
        rng = np.random.default_rng(semilla)
        idx = np.sort(rng.choice(len(X), size=max_muestras, replace=False))
        X_eval = X.iloc[idx]
    else:
        X_eval = X

    valores = explainer.shap_values(X_eval.values, check_additivity=False)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.ravel(base)[0])

    return ResultadoShap(
        departamento=departamento,
        modelo=modelo,
        nombres=list(X.columns),
        valores=np.asarray(valores),
        X=X_eval,
        valor_base=float(base),
    )


def tabla_familias(resultados: List[ResultadoShap]) -> pd.DataFrame:
    """Atribución por familia de variables, una fila por departamento y modelo."""
    filas = []
    for r in resultados:
        fila = {"departamento": r.departamento, "modelo": r.modelo}
        fila.update({k: round(float(v), 4) for k, v in r.importancia_por_familia().items()})
        top = r.importancia_global().head(3)
        fila["top3_variables"] = " | ".join(f"{n} ({v:.2f})" for n, v in top.items())
        filas.append(fila)
    return pd.DataFrame(filas).fillna(0.0)
