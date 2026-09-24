"""
Selector automático de modelos de pronóstico a partir del perfil de la serie.

Componentes:
    profiler.py  — extrae las metafeatures que describen una serie semanal.
    rules.py     — reglas de decisión perfil -> familia de modelo (selector v1).
    shap_utils.py— explicabilidad SHAP de los modelos de árboles (RF / XGBoost).
    evaluate.py  — mide el acierto del selector contra el ganador real.
"""
