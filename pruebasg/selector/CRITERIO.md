# Criterio de selección automática de modelo — versión 1

Documentación del criterio que, a partir del **perfil estadístico** de una serie
semanal de notificación obligatoria, determina qué familia de modelo de
pronóstico corresponde emplear.

El criterio no usa el nombre del departamento ni el de la enfermedad. Sus
entradas son exclusivamente metafeatures de la serie, de modo que es aplicable a
cualquier otra serie de vigilancia epidemiológica: la neumonía es el banco de
pruebas, no el objeto del criterio.

---

## 1. Por qué el criterio no persigue el MAE mínimo

La evidencia experimental (11 departamentos × 5 familias, walk-forward de
183–184 orígenes, prueba t pareada sobre los errores por origen con α = 0,05)
muestra que **en 10 de los 11 departamentos SARIMA, Random Forest y XGBoost
producen errores estadísticamente indistinguibles entre sí**. Solo en Lima
existe un ganador único.

Un selector entrenado para acertar cuál de los tres tiene el MAE mínimo estaría
ajustándose al ruido del estimador. Las decisiones que sí son distinguibles, y
las que el criterio persigue, son tres:

1. **Qué familia descartar sin evaluarla.** El aprendizaje profundo es
   significativamente peor en 10 de 11 departamentos, con un exceso de error
   medio del 50 %.
2. **Cuándo basta el modelo barato e interpretable.** Donde SARIMA pertenece al
   conjunto empatado, recomendarlo no cuesta exactitud medible y aporta
   intervalos de predicción y parámetros interpretables.
3. **Cuándo se paga la flexibilidad.** Solo donde el volumen es alto el modelo
   de árboles potenciados supera significativamente al estadístico
   (Lima +17,1 %, Piura +12,6 % frente a SARIMA).

## 2. Entradas del criterio

Producidas por [`profiler.py`](profiler.py):

| Metafeature | Símbolo | Uso en el criterio |
|---|---|---|
| Volumen medio semanal | `media_casos` | Umbrales de 10 y 40 casos/semana |
| Fuerza estacional (STL) | `fuerza_estacional` (F_S) | Umbral de 0,60 |
| Longitud de la ventana de entrenamiento | `train_size` | Descarte del modelo recurrente |

El perfilador calcula además estacionariedad (ADF y KPSS), fuerza de tendencia,
autocorrelaciones en los rezagos 1, 4 y 52, entropía espectral, índice de
dispersión, proporción de ceros, estabilidad y cambios de nivel. Estas no
intervienen en la versión 1 de las reglas, pero forman el vector completo sobre
el que se entrenará el clasificador de la versión 2.

## 3. Las reglas

Se evalúan en orden; la primera que se cumple decide.

| Id | Condición | Recomienda | Fundamento |
|---|---|---|---|
| **R0** | secuencias de entrenamiento < parámetros de la red | *descarta* LSTM | Sobreparametrización verificable antes de entrenar: 520 semanas con retrospección 52 producen ~465 secuencias frente a 17 076 parámetros (razón 1:37). |
| **R1** | `media_casos ≥ 40` y `F_S < 0,60` | XGBoost | Con ese volumen el ruido relativo de Poisson (1/√media) cae por debajo del 16 %: la estructura no lineal es señal aprovechable y no ruido de notificación. |
| **R2** | `F_S ≥ 0,60` | SARIMA con Fourier | La componente estacional explica la mayor parte de la varianza; la representación de Fourier la captura con 20 parámetros y entrega intervalos de predicción. |
| **R3** | `media_casos < 10` | SARIMA con Fourier | Ruido relativo de Poisson superior al 32 %: las variables de rezago están dominadas por la variabilidad de notificación y los modelos tabulares sobreajustan. |
| **R4** | caso restante | Random Forest | Volumen intermedio y estacionalidad moderada; es la alternativa más estable del conjunto empatado, por ser la menos sensible al ajuste de hiperparámetros. |

Los umbrales de 10 y 40 casos/semana no son empíricos: se derivan del ruido
relativo de un conteo de Poisson, que vale 1/√λ. El de 0,60 en fuerza estacional
proviene de la literatura de descomposición (Wang, Smith e Hyndman, 2006).

## 4. Cómo se evalúa el criterio

No por acierto del mínimo, sino por tres cantidades:

- **Cobertura** — fracción de series en las que la recomendación cae dentro del
  conjunto estadísticamente empatado con el mejor modelo. Es la métrica
  principal.
- **Regret** — exceso de MAE de la recomendación respecto del mejor modelo, en
  porcentaje.
- **Ahorro de cómputo** — costo de entrenar solo el modelo recomendado frente a
  evaluar las cuatro familias.

### Resultados

| Conjunto | Cobertura | Regret medio | Ahorro de cómputo |
|---|---|---|---|
| **En muestra** (11 departamentos con los que se construyeron las reglas) | 11/11 (100 %) | 1,80 % | 81,9 % |
| **Fuera de muestra** (14 departamentos restantes) | 13/14 (92,9 %) | 3,87 % | 81,9 % |

La tabla completa y el contraste con estrategias de modelo fijo están en
`pruebasg/informe_selector.html`.

## 5. Lo que el criterio NO demuestra

**Fuera de muestra, el criterio de reglas discretas no supera a una estrategia de
modelo fijo.** Sobre los mismos 14 departamentos:

| Estrategia | Cobertura | Regret medio | Regret máximo |
|---|---|---|---|
| siempre RandomForest | 14/14 | 1,70 % | 7,23 % |
| siempre XGBoost | 14/14 | 2,33 % | 6,85 % |
| **selector v1** | **13/14** | **3,87 %** | **23,31 %** |
| siempre SARIMA | 10/14 | 8,14 % | 23,31 % |
| siempre LSTM | 0/14 | 44,05 % | 80,13 % |

Presentar el selector como una mejora de exactitud sobre una estrategia fija **no
es sostenible con esta evidencia y no debe hacerse.**

Lo que sí sostiene la evidencia:

1. **El descarte del aprendizaje profundo (R0) es correcto y valioso.** LSTM queda
   fuera del conjunto empatado en 24 de las 25 series, con un exceso de error del
   44–50 %. Descartarlo a partir del perfil, antes de entrenarlo, evita la familia
   más costosa sin pérdida alguna.
2. **El criterio identifica dónde basta el modelo interpretable.** SARIMA
   pertenece al conjunto empatado en 19 de las 25 series; recomendarlo allí aporta
   intervalos de predicción y parámetros interpretables sin pérdida medible.
3. **El procedimiento de evaluación es reutilizable.** El conjunto estadísticamente
   empatado, la cobertura, el exceso de error y el ahorro de cómputo constituyen un
   marco de evaluación aplicable a cualquier criterio de selección, con
   independencia de que estas reglas concretas resulten o no las definitivas.

### Dónde falla, y por qué

El único fallo fuera de muestra es **Moquegua**: la regla R3 (volumen < 10 casos
semanales → SARIMA) recomienda el modelo estadístico donde el bosque aleatorio es
significativamente mejor (1,19 frente a 0,96 casos de MAE, p = 0,0004). La misma
regla acierta en las otras tres series de volumen comparable —Tacna 2,0, Tumbes
3,1 y Madre de Dios 4,7 casos semanales—, donde SARIMA sí pertenece al conjunto
empatado. **El volumen por sí solo no discrimina el caso que falla.**

Debe consignarse además que ese 23,3 % de exceso relativo corresponde a **0,22
casos por semana** en términos absolutos. El exceso relativo se infla cuando el
error de base ronda el caso semanal, por el mismo efecto de escala que afecta al
MAPE. Esto no anula el fallo, pero indica que el exceso relativo no debe ser la
única métrica de evaluación del criterio.

## 6. Verificación mediante SHAP

La atribución SHAP de los modelos de árboles ([`shap_utils.py`](shap_utils.py))
se emplea como instrumento de verificación del mecanismo, no como capa de
interpretabilidad añadida.

- **Confirma el mecanismo del volumen, sobre el que se apoyan R1 y R3.** La
  participación de la inercia de corto plazo (rezagos
  ≤ 4 semanas) correlaciona con el logaritmo del volumen en r = +0,87
  (Random Forest) y r = +0,80 (XGBoost); las ventanas móviles y el calendario
  correlacionan negativamente (−0,65 a −0,81). A mayor volumen, el modelo
  desplaza su atribución desde los agregados suavizados hacia el rezago
  inmediato, porque éste deja de estar dominado por el ruido de Poisson.
  El mecanismo queda así verificado; lo que la validación fuera de muestra
  muestra insuficiente no es el mecanismo sino el **umbral discreto** con que R3
  lo traduce en decisión (véase el fallo de Moquegua en la sección 5).
- **No confirma R2 por la vía esperada.** Frente a la fuerza estacional todas
  las correlaciones son casi nulas (|r| < 0,25): los árboles no incrementan su
  atribución al calendario en las series de estacionalidad fuerte. La lectura
  que refuerza la regla: si no explotan la estructura estacional aun cuando ésta
  domina la varianza, el modelo con estacionalidad explícita es el adecuado
  precisamente porque los otros no la capturan.
- **Hallazgo concreto.** La atribución de XGBoost al rezago anual es exactamente
  cero en los once departamentos, porque su configuración de variables
  (rezagos 1, 2, 4, 8, 13) no incluye el rezago 52. La limitación es del
  conjunto de características, no del algoritmo. Corregirla es trabajo inmediato.

## 7. Reproducir

```bash
python pruebasg/run_profiles.py          # perfila los 25 departamentos
python pruebasg/run_grid_selector.py     # walk-forward de la muestra estratificada
python pruebasg/run_selector.py          # aplica el criterio y lo evalúa
python pruebasg/run_shap.py              # atribución SHAP y contraste con el perfil
python pruebasg/build_selector_report.py # informe HTML autónomo
```

`run_grid_selector.py` es reanudable: omite los experimentos cuyo snapshot ya
existe, salvo `--force`.

## 8. Pendiente para la versión 2

1. Reemplazar las reglas discretas por un clasificador entrenado sobre el vector
   completo de metafeatures, con validación cruzada dejando un departamento fuera.
   La evidencia fuera de muestra indica que las reglas discretas no bastan: R3
   falla en Moquegua y el volumen por sí solo no separa ese caso de Tacna, Tumbes
   y Madre de Dios, donde la misma regla acierta. Candidatas a condición adicional
   sobre las que el clasificador debería discriminar: proporción de ceros, entropía
   espectral, cambio de nivel y longitud efectiva de la serie.
2. Reportar el exceso de error en casos además de en porcentaje, y fijar un umbral
   de materialidad por debajo del cual una diferencia no se considera fallo del
   criterio aunque sea estadísticamente significativa.
3. Validar en el grupo de edad de 60 años o más y, si es posible, en otra
   enfermedad de notificación obligatoria — es la prueba real de transferibilidad,
   y el único conjunto verdaderamente no usado que queda disponible.
4. Incorporar el rezago 52 al conjunto de características de XGBoost y repetir
   la comparación.
5. Extender la evaluación a horizontes mayores que una semana: la degradación
   del error recursivo penaliza de forma distinta a cada familia.
