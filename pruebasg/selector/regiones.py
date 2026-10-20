"""
Región natural predominante de cada departamento y la muestra estratificada
usada en la evaluación del selector.

La región natural NO entra en el selector como variable de decisión: entra como
variable de ESTRATIFICACIÓN, para garantizar que la muestra de departamentos
cubra perfiles de serie distintos (costa urbana de alto volumen, sierra con
estacionalidad marcada por friaje, selva con estacionalidad más plana). El
selector decide únicamente con el perfil estadístico de la serie, de modo que
el criterio siga siendo aplicable a series de otras enfermedades o países.

Departamentos que abarcan más de una región natural (ANCASH, AREQUIPA, HUANUCO,
LA LIBERTAD, PIURA, CAJAMARCA) se asignan a la región donde se concentra la
mayor parte de su población.
"""

REGION_NATURAL = {
    # Costa
    "TUMBES": "costa",
    "PIURA": "costa",
    "LAMBAYEQUE": "costa",
    "LA LIBERTAD": "costa",
    "LIMA": "costa",
    "CALLAO": "costa",
    "ICA": "costa",
    "AREQUIPA": "costa",
    "MOQUEGUA": "costa",
    "TACNA": "costa",
    # Sierra
    "ANCASH": "sierra",
    "CAJAMARCA": "sierra",
    "PASCO": "sierra",
    "JUNIN": "sierra",
    "HUANCAVELICA": "sierra",
    "AYACUCHO": "sierra",
    "APURIMAC": "sierra",
    "CUSCO": "sierra",
    "PUNO": "sierra",
    # Selva
    "AMAZONAS": "selva",
    "LORETO": "selva",
    "SAN MARTIN": "selva",
    "UCAYALI": "selva",
    "MADRE DE DIOS": "selva",
    "HUANUCO": "selva",
}

# Muestra estratificada por región natural × volumen de casos.
#
# Criterio de armado: cubrir las tres regiones naturales y, dentro de cada una,
# el rango de volumen semanal disponible. MADRE DE DIOS se incluye a propósito
# pese a su bajo volumen (~5 casos/semana): es el único departamento que ocupa
# la categoría de volumen bajo del perfilador, y sin él esa rama del selector
# quedaría sin evidencia que la respalde.
MUESTRA_ESTRATIFICADA = [
    # costa
    "LIMA",           # volumen alto     (~157 casos/sem)
    "PIURA",          # volumen alto     (~47)
    "LA LIBERTAD",    # volumen medio    (~25)
    "ICA",            # volumen medio    (~12)
    # sierra
    "CUSCO",          # volumen medio    (~38)
    "PUNO",           # volumen medio    (~31)
    "HUANCAVELICA",   # volumen medio    (~17)
    # selva
    "LORETO",         # volumen alto     (~59)
    "UCAYALI",        # volumen medio    (~30)
    "AMAZONAS",       # volumen medio    (~17)
    "MADRE DE DIOS",  # volumen bajo     (~5)
]


def region_de(departamento: str) -> str:
    """Región natural predominante; 'desconocida' si el nombre no está mapeado."""
    return REGION_NATURAL.get(departamento.strip().upper(), "desconocida")
