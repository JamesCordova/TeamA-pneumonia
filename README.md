# Modelado de Neumonía — Forecasting de Casos por Departamento

Pipeline de series temporales para pronosticar casos semanales de neumonía en los 25 departamentos del Perú, segmentados por grupo de edad (**menores de 5 años** y **adultos mayores de 60**). Utiliza datos IRAS del MINSA (2000–2023).

## Modelos disponibles

| Modelo | Tipo |
|---|---|
| Naive | Baseline (último valor observado) |
| SeasonalNaive | Baseline estacional (semana equivalente del año anterior) |
| HoltWinters | Suavización exponencial triple |
| SARIMA | Auto-ARIMA estacional (`pmdarima`) |
| RandomForest | Ensamble con features de calendario y lag |
| XGBoost | Gradient boosting con features de calendario y lag |
| Prophet | Modelo aditivo/multiplicativo de Meta para series temporales (con feriados de Perú) |
| LSTM | Red neuronal recurrente (RNN) Long Short-Term Memory para dependencias secuenciales |
| GRU | Red neuronal recurrente (RNN) Gated Recurrent Unit, alternativa más ligera a LSTM |

## Flujo de trabajo

```
prepare_data → train_* → run_walkforward → plot_forecasting / compare_models
```

1. **Preparar datos** — descarga + limpieza (rellena 3 saltos ISO semana 53)
2. **Entrenar** — split clásico train/val/test por modelo (No debería ser necesario)
3. **Walk-forward** — validación rolling-origin (horizonte configurable)
4. **Visualizar y comparar** — gráficos de backtest y tabla comparativa de métricas

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/JamesCordova/Modelado-Pneumonia
cd Modelado-Pneumonia

# 2. Crear y activar entorno virtual
python -3.12 -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env        # editar si es necesario (las rutas por defecto funcionan)
```

## Uso

### 1. Preparar datos

```bash
# Descarga desde la base de datos y genera data/processed/iras_weekly_clean.csv
python scripts/prepare_data.py

# Si el raw ya existe, solo procesar
python scripts/prepare_data.py --no_download

# Forzar sobreescritura del archivo procesado
python scripts/prepare_data.py --force
```

> **Nota:** Los años 2004, 2009 y 2015 tienen un salto de 14 días por la semana ISO 53.
> `prepare_data.py` los rellena automáticamente por interpolación lineal.

### 2. Entrenar modelos (split clásico)(No debería ejecutarse para el análisis)

```bash
python scripts/train_baselines.py   --department AMAZONAS
python scripts/train_sarima.py      --department AMAZONAS
python scripts/train_random_forest.py --department AMAZONAS
python scripts/train_xgboost.py     --department AMAZONAS
python scripts/train_prophet.py      --department AMAZONAS

# Grupo de edad (default: under5)
python scripts/train_sarima.py --department LIMA --age_group 60plus

# Sobreescribir hiperparámetros en Prophet
python scripts/train_prophet.py --department LIMA --age_group 60plus --changepoint_prior_scale 0.1
```

> **Subregistro:** MOQUEGUA, TACNA y TUMBES presentan ceros sistemáticos antes de 2008/2009.
> Se recomienda usar `--start_year 2009` (MOQUEGUA) o `--start_year 2008` (TACNA, TUMBES).

### 2.1 Optimización automática de hiperparámetros (Tuning)

Optimiza hiperparámetros para modelos de predicción (`RandomForest`, `XGBoost`, `Prophet`, `LSTM` y `GRU`) evaluando diferentes combinaciones de parámetros en el set de validación.

> **Nota sobre RNN (LSTM/GRU):** Para los modelos basados en redes neuronales recurrentes, la sintonización establece el valor por defecto de `--n_iter` en 10 combinaciones para mantener un tiempo de ejecución eficiente. Dado que estos modelos requieren TensorFlow, se recomienda ejecutarlos dentro del entorno de Docker del proyecto.

```bash
# Búsqueda aleatoria para RandomForest (default: 20 iteraciones)
python scripts/tune_models.py --department AMAZONAS --model RandomForest --n_iter 15

# Búsqueda por grilla para XGBoost
python scripts/tune_models.py --department AMAZONAS --model XGBoost --search_method grid

# Sintonización para Prophet (se guardarán las recomendaciones para config.py de Prophet)
python scripts/tune_models.py --department LIMA --model Prophet --n_iter 10

# Sintonización para LSTM (n_iter=10 por defecto para RNNs)
python scripts/tune_models.py --department LIMA --model LSTM
```

### 3. Validación walk-forward (rolling-origin)

```bash
# Ejemplo básico
python scripts/run_walkforward.py --department AMAZONAS --model SARIMA

# Personalizar horizonte, paso y ventana
python scripts/run_walkforward.py \
    --department LIMA --age_group 60plus \
    --model XGBoost \
    --horizon 8 --step 4 \
    --window_type expanding \
    --train_size 260

# Ejecutar para todos los departamentos
python scripts/run_walkforward.py --all --model Naive --horizon 4

# Cortar años con subregistro
python scripts/run_walkforward.py --department MOQUEGUA --model SARIMA --start_year 2009

# LSTM / GRU con hiperparámetros personalizados
python scripts/run_walkforward.py --department UCAYALI --model LSTM \
    --train_size 260 --horizon 4 --step 4 --refit_every 1 \
    --lookback 52 --units_1 64 --units_2 32 --epochs 80

python scripts/run_walkforward.py --all --model GRU --age_group 60plus \
    --train_size 260 --horizon 4 --refit_every 1 --smooth_window 1

# Prophet en walk-forward
python scripts/run_walkforward.py --department LIMA --model Prophet --horizon 4
```

Modelos disponibles en `--model`: `SARIMA`, `RandomForest`, `XGBoost`, `LSTM`, `GRU`, `HoltWinters`, `SeasonalNaive`, `Naive`, `Prophet`

### 4. Graficar resultados

```bash
# Plot de backtest (walk-forward)
python scripts/plot_forecasting.py --department AMAZONAS --plot_type backtest

# Plot clásico (train/val/test)
python scripts/plot_forecasting.py --department AMAZONAS --plot_type classic

# Filtrar por año o modelos específicos
python scripts/plot_forecasting.py --department AMAZONAS --plot_type backtest \
    --year 2022 --models SARIMA XGBoost
```

### 5. Comparar modelos

```bash
# Tabla + figura con MAE (default)
python scripts/compare_models.py --department AMAZONAS

# Cambiar métrica del gráfico
python scripts/compare_models.py --department AMAZONAS --metric rmse
python scripts/compare_models.py --department AMAZONAS --metric smape
python scripts/compare_models.py --department AMAZONAS --metric mda

# Horizontes personalizados
python scripts/compare_models.py --department AMAZONAS --horizons 1 4
```

Genera en `reports/{DEPT}/{AGE_GROUP}/`:
- `model_comparison.csv` — tabla MAE · RMSE · SMAPE · MDA por horizonte
- `model_comparison_{metric}.png` — barras agrupadas h=1 vs h=4 + líneas de evolución

## Estructura del proyecto

```
Modelado-Pneumonia/
├── .env.example                 # Plantilla de configuración
├── requirements.txt
│
├── pneumonia/                   # Paquete principal
│   ├── config.py                # Rutas, splits, parámetros SARIMA
│   ├── utils.py                 # Logger
│   ├── data/                    # Descarga y carga de datos IRAS
│   │   ├── download_raw_data.py
│   │   └── load_data.py
│   ├── models/
│   │   ├── base.py              # Clase base abstracta
│   │   ├── utils.py             # get_departmental_data, temporal_split
│   │   ├── baselines/           # Naive, SeasonalNaive, HoltWinters
│   │   ├── sarima/              # SARIMAModel (pmdarima)
│   │   ├── ml/                  # RandomForestModel, XGBoostModel, EnsembleModel
│   │   ├── prophet/             # ProphetModel
│   │   └── rnn/                 # LSTMModel, GRUModel (Redes Neuronales Recurrentes)
│   ├── evaluation/
│   │   ├── metrics.py           # MAE, RMSE, SMAPE, MDA, MAPE
│   │   └── walkforward.py       # WalkForwardValidator
│   ├── pipelines/               # Pipelines train/val/test por modelo
│   ├── visualization/
│   │   ├── classic_plot.py      # Gráfico clásico (train/val/test)
│   │   ├── backtest_plot.py     # Gráfico de backtest walk-forward
│   │   ├── comparison_plot.py   # Figura comparativa de modelos
│   │   └── persistence.py       # Lectura/escritura de CSVs de predicciones
│   └── eda/                     # Análisis exploratorio
│
├── scripts/                     # Puntos de entrada CLI
│   ├── prepare_data.py          # Descargar + limpiar datos
│   ├── train_baselines.py
│   ├── train_sarima.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_prophet.py         # Entrenar modelo Prophet
│   ├── tune_models.py           # Optimización automática de hiperparámetros (ML)
│   ├── run_walkforward.py       # Walk-forward para cualquier modelo
│   ├── plot_forecasting.py      # Visualización de resultados
│   └── compare_models.py        # Comparación cruzada de modelos
│
├── data/
│   ├── raw/                     # iras_data_raw.csv (fuente original)
│   └── processed/               # iras_weekly_clean.csv (7D regular, gaps rellenos)
│
├── models/                      # Modelos entrenados (.pkl)
└── reports/
    └── {DEPT}/{AGE_GROUP}/
        ├── {model}_predictions.csv           # Predicciones (train/val/test/backtest)
        ├── {model}_walkforward_metrics.json  # MAE·RMSE·SMAPE·MDA por horizonte
        ├── backtest_plot.png
        └── model_comparison_{metric}.png
```

## Variables de entorno

Las rutas y parámetros se configuran en `.env`. Todos tienen valores por defecto funcionales:

| Variable | Default | Descripción |
|---|---|---|
| `DATA_RAW_PATH` | `data/raw/` | Datos originales |
| `DATA_PROCESSED_PATH` | `data/processed/` | Datos limpios |
| `MODEL_STORAGE_PATH` | `models/` | Modelos serializados |
| `REPORTS_PATH` | `reports/` | Predicciones y figuras |
| `TEMPORAL_SPLIT_STRATEGY` | `dynamic` | `dynamic` (ratios) o `years` (años fijos) |
| `TRAIN_RATIO` | `0.8` | Fracción de datos para entrenamiento |
| `USE_AUTO_ARIMA` | `True` | Búsqueda automática de orden SARIMA |
| `SARIMA_STEPWISE` | `True` | Búsqueda stepwise (más rápida) |
| `RANDOM_SEED` | `42` | Semilla de reproducibilidad |

## Métricas de evaluación

| Métrica | Descripción |
|---|---|
| MAE | Error absoluto medio (casos) |
| RMSE | Raíz del error cuadrático medio (casos) — penaliza outliers |
| SMAPE | Error porcentual simétrico — robusto ante conteos bajos |
| MDA | Dirección correcta del cambio (%) |

> MAPE se calcula internamente pero **no se usa** en comparaciones: es inestable cuando los conteos se acercan a cero.

## Contribuir

1. Crear rama: `git checkout -b feature/nombre`
2. Commit con [Conventional Commits](https://www.conventionalcommits.org/): `git commit -m "feat: ..."`
3. Abrir Pull Request hacia `main`
