# Guía de Inicialización y Ejecución del Proyecto

Esta guía contiene los pasos detallados para configurar el entorno virtual con **Conda**, instalar las dependencias necesarias, conectar la base de datos externa (solo lectura) y ejecutar el pipeline completo de modelado de neumonía.

---

## ⚙️ 1. Configuración del Entorno Virtual (Conda)

Abre tu terminal en la raíz del proyecto y ejecuta:

```bash
# 1. Crear el entorno virtual con Python 3.12 (versión recomendada)
conda create -y -n pneumonia python=3.12

# 2. Activar el entorno virtual
conda activate pneumonia

# 3. Instalar todas las dependencias
pip install -r requirements.txt
```

---

## 🗄️ 2. Conectar tu Base de Datos Existente

El código **solo lee** los datos de tu base de datos y genera archivos locales (no realiza modificaciones). 

1. Copia el archivo de ejemplo a tu archivo de configuración real:
   ```bash
   cp .env.example .env
   ```
2. Abre el archivo `.env` en tu editor de código y configura la URL de conexión a tu PostgreSQL en la variable `DATABASE_URL`:
   ```ini
   DATABASE_URL=postgresql://usuario:contraseña@servidor:5432/nombre_base_datos
   ```

---

## 🚀 3. Ejecución del Pipeline

Sigue este flujo secuencial para ejecutar los modelos de predicción:

### Paso A: Descarga y Preparación de Datos
Este comando extraerá los datos de tu base de datos externa y creará los archivos CSV locales necesarios en `data/raw/` y `data/processed/`:
```bash
python scripts/prepare_data.py
```
*(Al finalizar esta ejecución, verás una lista de los departamentos que se detectaron en tu base de datos).*

### Paso B: Entrenar Modelos
Entrena los modelos predictivos. Puedes entrenar un departamento individual o una lista de departamentos separados por espacios o comas:

```bash
# Entrenar modelo SARIMA para Amazonas y Lima
python scripts/train_sarima.py --department AMAZONAS LIMA --age_group under5

# Entrenar RandomForest con parámetros personalizados
python scripts/train_random_forest.py --department AMAZONAS LIMA --n_estimators 150 --max_depth 10 --lags 1 2 4 --windows 4 13
```

### Paso C: Validación Walk-Forward (Backtesting)
Evalúa el rendimiento predictivo de los modelos entrenados usando validación cruzada temporal:
```bash
# Ejecutar validación para SARIMA
python scripts/run_walkforward.py --department AMAZONAS LIMA --model SARIMA

# Ejecutar validación para RandomForest
python scripts/run_walkforward.py --department AMAZONAS LIMA --model RandomForest
```

### Paso D: Comparar y Visualizar Resultados
Analiza el error de cada modelo y genera los reportes correspondientes:

```bash
# Generar tabla de comparación de métricas (MAE, RMSE, SMAPE, MDA)
python scripts/compare_models.py --department AMAZONAS LIMA --metric smape

# Generar gráficos de predicción y backtest
python scripts/plot_forecasting.py --department AMAZONAS LIMA --plot both
```
*(Los gráficos e informes en formato JSON/CSV se guardarán dentro de la carpeta `reports/`).*

---

## 🛠️ 4. Parámetros Opcionales y Personalización

Puedes ajustar el comportamiento de los modelos y las evaluaciones a través de los siguientes flags en la línea de comandos:

### Parámetros Generales
*   `--age_group` / `-g`: Grupo de edad. Opciones: `under5` (menores de 5 años, por defecto) o `60plus` (adultos mayores de 60 años).
*   `--split_strategy` / `-s`: Estrategia de partición temporal. Opciones: `dynamic` (porcentajes de datos) o `years` (años calendarios fijos).
*   `--verbose` / `-v`: Activa la salida de logs en nivel DEBUG.
*   `--quiet` / `-q`: Silencia la salida informativa para acelerar los scripts.

### Hiperparámetros de Modelos de ML (`RandomForest` / `XGBoost`)
Estos parámetros se pueden pasar tanto a los scripts de entrenamiento (`train_*`) como al script de validación walk-forward (`run_walkforward.py`):
*   `--n_estimators`: Número de estimadores/árboles en el ensamble.
*   `--max_depth`: Profundidad máxima permitida para los árboles.
*   `--learning_rate`: [Solo XGBoost] Tasa de aprendizaje o eta de reducción.
*   `--subsample`: [Solo XGBoost] Fracción de filas para muestrear en cada árbol.
*   `--colsample_bytree`: [Solo XGBoost] Fracción de columnas para muestrear por árbol.
*   `--min_samples_leaf`: [Solo RandomForest] Mínimo de muestras por nodo hoja.
*   `--min_samples_split`: [Solo RandomForest] Mínimo de muestras para dividir un nodo.
*   `--max_features`: [Solo RandomForest] Atributos a considerar por nodo (`sqrt`, `log2`).
*   `--lags`: Lista de rezagos (lags) temporales a incluir como variables (e.g. `--lags 1 2 4 8 13`).
*   `--windows`: Lista de ventanas móviles para promedios históricos (e.g. `--windows 4 13 26`).

### Parámetros de SARIMA (`train_sarima.py` / `run_walkforward.py`)
*   `--sarima_order`: Orden no estacional manual `(p, d, q)` (e.g. `--sarima_order 2 1 1`).
*   `--n_fourier_terms`: Cantidad de pares seno/coseno de Fourier para modelar la estacionalidad (por defecto: `6`).
*   `--no_fourier`: Fuerza el uso de estacionalidad SARIMA clásica en lugar de regresores de Fourier.
*   `--fourier`: Fuerza el modelado estacional con Fourier (ignora la configuración por defecto).
*   `--use_auto_arima`: Activa la búsqueda automática de parámetros vía `pmdarima` (ignora valores por defecto).
*   `--no_auto_arima`: Fuerza el uso del orden manual o el configurado por defecto.

### Parámetros de Validación Walk-Forward (`run_walkforward.py`)
*   `--train_size`: Tamaño de la ventana inicial de entrenamiento en semanas (por defecto `520` ≈ 10 años).
*   `--horizon`: Horizonte de pronóstico a evaluar por paso en semanas (por defecto `4`).
*   `--step`: Desplazamiento temporal del origen de pronóstico en semanas (por defecto `4`).
*   `--window_type`: Tipo de ventana. Opciones: `sliding` (ventana deslizante fija) o `expanding` (ventana expansiva incremental).
*   `--refit_every`: Frecuencia con la que se re-entrena el modelo (en pasos). Configura `0` para entrenar una sola vez al inicio, o `1` para cada paso.
*   `--start_year`: Año a partir del cual entrenar (útil para excluir datos incompletos tempranos de departamentos específicos).

---

## 📊 5. Métricas de Evaluación Disponibles y Márgenes de Calidad

El pipeline calcula y reporta las siguientes métricas de precisión, ajuste y sesgo. A continuación se detallan sus significados y los **márgenes o umbrales de referencia** para evaluar si un modelo es de buena calidad:

1.  **MAE (Error Absoluto Medio)**: Mide la magnitud promedio de los errores de predicción en unidades de casos de neumonía.
    *   *Margen*: Debe ser lo más bajo posible. Se evalúa de manera relativa comparándolo con el volumen total de casos del departamento.
2.  **RMSE (Raíz del Error Cuadrático Medio)**: Mide el error promedio penalizando con mayor fuerza las predicciones muy alejadas (errores grandes).
    *   *Margen*: Al igual que MAE, debe ser lo más bajo posible. Si RMSE es mucho mayor que el MAE, indica presencia de errores muy grandes en semanas específicas.
3.  **ME (Error Medio / Sesgo)**: Evalúa si el modelo subestima o sobreestima de forma sistemática.
    *   *Valor Óptimo*: `0.0` (modelo perfectamente insesgado).
    *   *Margen*: 
        *   **ME < 0**: El modelo tiende a **sobreestimar** (pronostica más casos de los reales).
        *   **ME > 0**: El modelo tiende a **subestimar** (pronostica menos casos de los reales).
        *   Un sesgo de ±1 a ±3 casos se considera aceptable dependiendo del tamaño del departamento.
4.  **R2 (Coeficiente de Determinación)**: Representa la proporción de la varianza histórica que el modelo logra capturar.
    *   *Rango*: $-\infty$ a `1.0`.
    *   *Margen*:
        *   **R2 >= 0.70**: Ajuste **excelente** (muy alta capacidad predictiva).
        *   **R2 entre 0.40 y 0.70**: Ajuste **bueno a moderado** (habitual en series ruidosas de salud pública).
        *   **R2 < 0.40**: Ajuste **pobre** (baja confianza en las predicciones).
        *   **R2 <= 0.00**: El modelo predice peor que simplemente usar el promedio histórico de la serie.
5.  **MASE (Error Absoluto Escalado Medio)**: Compara el error del modelo frente a una predicción ingenua (naive) estacional. ¡Es la métrica clave de utilidad!
    *   *Rango*: `0.0` a $\infty$.
    *   *Margen*:
        *   **MASE < 1.0**: **El modelo es útil** (es más preciso que simplemente repetir el valor del año pasado).
        *   **MASE < 0.50**: Ajuste **excepcional** (muy superior al baseline).
        *   **MASE >= 1.0**: El modelo no aporta valor predictivo frente a la estimación más simple posible.
6.  **SMAPE (Error Porcentual Absoluto Medio Simétrico)**: Mide el error promedio en formato porcentual (de 0% a 200%).
    *   *Margen*:
        *   **SMAPE < 10%**: Precisión **excelente**.
        *   **SMAPE entre 10% y 25%**: Precisión **buena a aceptable** (típica en epidemiología debido a fluctuaciones estacionales).
        *   **SMAPE > 25%**: Precisión **baja**.
7.  **MDA (Precisión Direccional Media)**: Mide el porcentaje de aciertos en predecir si los casos subirán o bajarán la semana siguiente.
    *   *Rango*: `0%` a `100%`.
    *   *Margen*:
        *   **MDA > 50%**: El modelo predice la dirección mejor que el azar (lanzar una moneda).
        *   **MDA >= 70%**: **Excelente** capacidad para detectar tendencias y alertas tempranas de brotes.

---

## 🗺️ 6. Departamentos de Perú Disponibles

El proyecto procesa los **25 departamentos oficiales** del Perú. Cuando ejecutes `scripts/prepare_data.py`, el sistema validará cuáles de ellos tienen suficiente volumen de datos histórico (mínimo 104 semanas) antes de habilitar su entrenamiento.

Lista completa de departamentos a seleccionar:
1.  `AMAZONAS`
2.  `ANCASH`
3.  `APURIMAC`
4.  `AREQUIPA`
5.  `AYACUCHO`
6.  `CAJAMARCA`
7.  `CALLAO`
8.  `CUSCO`
9.  `HUANCAVELICA`
10. `HUANUCO`
11. `ICA`
12. `JUNIN`
13. `LA LIBERTAD`
14. `LAMBAYEQUE`
15. `LIMA`
16. `LORETO`
17. `MADRE DE DIOS`
18. `MOQUEGUA` *(Se recomienda iniciar en 2008 usando `--start_year 2008`)*
19. `PASCO`
20. `PIURA`
21. `PUNO`
22. `SAN MARTIN`
23. `TACNA` *(Se recomienda iniciar en 2007 usando `--start_year 2007`)*
24. `TUMBES` *(Se recomienda iniciar en 2007 usando `--start_year 2007`)*
25. `UCAYALI`
