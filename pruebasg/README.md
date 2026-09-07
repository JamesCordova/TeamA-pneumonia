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
