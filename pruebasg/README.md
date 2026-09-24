# pruebasg — Interfaz de presentación de resultados

Mini-suite para **presentar la comparación de modelos** de pronóstico de neumonía:
ejecuta walk-forward con distintos modelos/parámetros, guarda cada corrida como un
*snapshot* independiente (para que variar parámetros del mismo modelo no se
sobrescriba) y genera un **dashboard HTML autónomo** con los comandos ejecutados,
las métricas y los gráficos de backtest.

> Carpeta local de trabajo. No forma parte del paquete `pneumonia`.

## Requisitos

- El entorno virtual del proyecto activado (trae pandas + matplotlib):
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- Datos ya preparados: `data/processed/iras_weekly_clean.csv` (ya existe).

## Uso (2 pasos)

```powershell
# 1) Ejecutar los experimentos (edita EXPERIMENTS en run_grid.py para cambiarlos)
python pruebasg/run_grid.py
#    Ver los comandos sin ejecutar:
python pruebasg/run_grid.py --dry-run

# 2) Generar el dashboard
python pruebasg/build_dashboard.py
#    Zoom del backtest a un solo año (más legible para presentar):
python pruebasg/build_dashboard.py --year 2022
```

Luego abre **`pruebasg/dashboard.html`** con doble clic. Es un único archivo
(gráficos incrustados), no necesita servidor ni internet.

## Archivos

| Archivo | Qué hace |
|---|---|
| `run_grid.py` | Corre cada experimento (`scripts/run_walkforward.py`) y guarda snapshot + comando exacto en `results/manifest.json`. |
| `build_dashboard.py` | Lee `results/` y arma `dashboard.html` (comandos, tablas de métricas, gráficos). |
| `_common.py` | Rutas y utilidades compartidas. |
| `results/` | Snapshots por experimento + `manifest.json`. |

## Añadir o cambiar experimentos

Edita la lista `EXPERIMENTS` en [`run_grid.py`](run_grid.py). Cada entrada:

```python
{
    "label": "XGBoost (tuned)",       # nombre que verás en el dashboard
    "department": "AMAZONAS",
    "model": "XGBoost",               # SARIMA, RandomForest, XGBoost, SeasonalNaive, Naive, HoltWinters...
    "args": ["--n_estimators", "500", "--learning_rate", "0.01"],
},
```

Para otro departamento, duplica las entradas cambiando `"department"`.

## Modelos y departamentos

La grilla (`run_grid.py`) usa el producto `DEPARTMENTS × MODELS`:

- **Departamentos**: `AMAZONAS`, `LIMA` (edita `DEPARTMENTS` para añadir más).
- **Modelos**: SeasonalNaive, RandomForest (×2 configs), XGBoost, SARIMA,
  **LSTM**, **GRU**, **Prophet**.

## Notas

- **LSTM / GRU** requieren `tensorflow` (`pip install tensorflow`). Ya instalado.
- **Prophet** requiere `prophet` **y** un backend CmdStan compilado (RTools + CmdStan
  en Windows). Con `prophet` sin CmdStan da `'Prophet' object has no attribute
  'stan_backend'`. Para habilitarlo: `python -m cmdstanpy.install_cmdstan --compiler`
  (necesita toolchain C++). Por eso Prophet está comentado en `MODELS` de `run_grid.py`.
- **SARIMA / Prophet** usan `--refit_every 13` y **LSTM / GRU** `--refit_every 52`
  para que el walk-forward sea tratable en tiempo (si no, reentrenan en cada paso;
  LSTM/GRU son lentos de entrenar).
- En las tablas: MAE / RMSE / SMAPE en **casos** (menor = mejor); MDA en **%**
  (mayor = mejor). Las celdas verdes marcan el mejor valor por métrica.

---

# Selector automático de modelos

Segunda línea de trabajo de esta carpeta, y componente de aporte de la tesis:
un criterio que, a partir del **perfil estadístico** de una serie semanal de
notificación obligatoria, determina qué familia de modelo corresponde emplear.
No usa el nombre del departamento ni el de la enfermedad.

El criterio completo, con su fundamento, sus reglas, su evaluación y sus
limitaciones declaradas, está en **[`selector/CRITERIO.md`](selector/CRITERIO.md)**.

## Flujo

```powershell
python pruebasg/run_profiles.py          # perfila los 25 departamentos
python pruebasg/run_grid_selector.py     # walk-forward sobre la muestra estratificada
python pruebasg/run_selector.py          # aplica el criterio y mide su acierto
python pruebasg/run_shap.py              # atribución SHAP y contraste con el perfil
python pruebasg/build_selector_report.py # informe HTML autónomo
```

Abre **`pruebasg/informe_selector.html`** con doble clic: archivo único, figuras
incrustadas, sin servidor ni conexión.

## Archivos

| Archivo | Qué hace |
|---|---|
| `selector/profiler.py` | Metafeatures de una serie: volumen, dispersión, estacionariedad (ADF/KPSS), fuerzas de tendencia y estacionalidad por STL, autocorrelaciones, entropía espectral, estabilidad, cambios de nivel. |
| `selector/regiones.py` | Región natural de cada departamento y la muestra estratificada. La región estratifica la muestra; **no** entra en la decisión. |
| `selector/rules.py` | El criterio: reglas R0–R4 sobre el perfil, con su justificación y el costo estimado de cada familia. |
| `selector/evaluate.py` | Comparación pareada por origen y **conjunto estadísticamente empatado** con el mejor modelo. |
| `selector/shap_utils.py` | Atribución SHAP de Random Forest y XGBoost, agregada por familia de variables. |
| `run_grid_selector.py` | Grilla walk-forward reanudable sobre la muestra estratificada (una configuración por familia). |
| `build_selector_report.py` | Informe HTML con perfiles, evidencia, criterio, evaluación, validación fuera de muestra y SHAP. |

## Lo que hay que saber antes de leer los resultados

En **10 de 11 departamentos SARIMA, Random Forest y XGBoost son
estadísticamente indistinguibles** (prueba t pareada sobre los errores por
origen, α = 0,05). Por eso el selector no se evalúa por acertar el MAE mínimo
—eso sería ajustarse al ruido— sino por **cobertura** (que la recomendación caiga
dentro del conjunto empatado), **regret** y **ahorro de cómputo**.

La estrategia «siempre XGBoost» iguala al selector en exactitud. La ventaja del
criterio está en identificar dónde basta el modelo interpretable y en descartar
el aprendizaje profundo antes de entrenarlo, no en reducir el error.
