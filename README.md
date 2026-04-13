# TeamA Pneumonia - Detección de Neumonía con Machine Learning

Proyecto de machine learning para el desarrollo de pipelines de modelado escalables con almacenamiento centralizado de datos y modelos en PostgreSQL. Enfocado en la detección automática de neumonía mediante técnicas de visión por computadora.

## Objetivos del Proyecto

- **Desarrollar pipelines de modelado escalables**: Crear arquitecturas robustas y modulares para entrenamiento, validación e inferencia de modelos
- **Gestionar datos de entrenamiento en PostgreSQL**: Cargar y procesar datos directamente desde la base de datos centralizada
- **Almacenar modelos/pesos de forma centralizada**: Utilizar PostgreSQL como repositorio central para versioning y gestión de artefactos de ML
- **Reproducibilidad**: Mantener tracking de experimentos y resultados
- **Modularidad**: Código reutilizable y bien estructurado para facilitar colaboración

## Requisitos Previos

- Python 3.9+
- PostgreSQL 12+ (para datos de entrenamiento y almacenamiento de modelos)
- Git

## Get Started

### 1. Clonar el Repositorio

```bash
git clone https://github.com/JamesCordova/TeamA-pneumonia
cd TeamA-pneumonia
```

### 2. Crear Virtual Environment

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno

Crear un archivo `.env` en la raíz del proyecto:

```
# Base de datos (datos de entrenamiento + modelos)
DATABASE_URL=postgresql://usuario:contraseña@localhost:5432/pneumonia
MODEL_STORAGE_PATH=models/
LOG_PATH=logs/
```

## Estructura del Proyecto

```
TeamA-pneumonia/
├── .env.example              # Plantilla de variables de entorno
├── .gitignore
├── README.md
├── requirements.txt          # Dependencias del proyecto
│
├── pneumonia/                # Código principal del proyecto
│   ├── __init__.py
│   ├── config.py             # Configuración general del proyecto
│   ├── data.py               # Funciones de carga de datos desde PostgreSQL
│   ├── utils.py              # Funciones utilitarias (logging, etc.)
│   └── pipelines/            # Pipelines de ML
│       ├── __init__.py
│       └── (pipelines disponibles)
│
├── data/                     # Almacenamiento de datos
│   ├── raw/                  # Datos sin procesar
│   ├── processed/            # Datos procesados
│   ├── interim/              # Datos intermedios
│   └── external/             # Datos externos
│
├── models/                   # Modelos entrenados
│
├── notebooks/                # Jupyter notebooks para análisis y experimentación
│   ├── prototype.ipynb       # Prototipo inicial
│   └── archive/              # Notebooks archivados
│
└── tests/                    # Tests unitarios
    ├── __init__.py
    └── test_models.py        # Tests para modelos
```

## Gestión de Datos

### Datos de Entrenamiento

Todos los datos de entrenamiento se almacenan en PostgreSQL. Los datos se cargan automáticamente desde la BD mediante el módulo `src/pneumonia/data.py`:

```python
from pneumonia.data import load_training_data

# Cargar datos de entrenamiento desde BD
df_train = load_training_data(table_name="training_data", limit=None)
```

### Estructura de Datos en BD

Cada tabla de datos en PostgreSQL contiene:
- Metadatos (fecha de carga, origen, versión)
- Características (features)
- Etiquetas (labels) para datos etiquetados

## Almacenamiento de Modelos

Todos los modelos y sus pesos se almacenan en PostgreSQL con:
- **Versionado**: Tracking de todas las versiones
- **Metadatos**: Información de entrenamiento, métricas, hiperparámetros
- **Trazabilidad**: Quién, cuándo y por qué se guardó cada modelo

## Uso (Por definir)

### Entrenamiento de Modelos (Por definir)

```bash
python scripts/train_model.py --config config.yaml
```

### Inferencia (Por definir)

```bash
python scripts/inference.py --model-id <model-id> --data data.csv
```

### Desarrollo y Experimentación

Usar los notebooks en `notebooks/` para exploración y prototipado. Mover código probado a módulos en `pneumonia/`.



## Contribuir

1. Crear una rama para tu feature: `git checkout -b feature/nombre-de-feature`
2. Hacer commit de cambios: `git commit -m "descripción"` (Conventional commits)
3. Push a la rama: `git push origin feature/nombre`
4. Abrir un Pull Request (Push a la rama main no es recomendado)
