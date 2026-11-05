# Segmentos de precio de departamentos en CABA

Proyecto integrador de Programación Avanzada en Ciencia de Datos.
Clasificación binaria de publicaciones de departamentos: segmento alto frente
a estándar. El trabajo queda local; no se inicializó ni se publicó un repositorio Git.

## Qué abrir

`proyecto_inmuebles.ipynb` es el documento principal: explica las decisiones,
ejecuta el entrenamiento y muestra los resultados. `inmuebles/*.py` contiene
funciones reutilizables; `ejecutar.py` permite ejecutar lo mismo sin Jupyter.
No hay dos implementaciones del entrenamiento.

`resultados/analisis_papermill.ipynb` contiene la ejecución completa con
Papermill 2.7.0: parámetros inyectados, entrenamiento, métricas, gráficos y
consultas SQLite. Todas las celdas finalizaron sin errores.

## Preparar y ejecutar

Desde esta carpeta, crear y activar un entorno con Python 3.14:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python ejecutar.py
```

La ejecución procesa el dataset completo, ajusta los modelos y genera
`resultados/departamentos_2020/`. En Windows se activa el entorno con
`.venv\Scripts\activate`. La corrida puede tardar varios minutos.

Para ejecutar el notebook completo con Papermill desde un kernel limpio,
inyectar los parámetros y guardar la versión ejecutada:

```bash
python -m papermill proyecto_inmuebles.ipynb resultados/analisis_papermill.ipynb -p run_id departamentos_2020 -p cuantil 0.75 -p semilla 42 --execution-timeout 1800
```

El notebook tiene una celda etiquetada `parameters`. Papermill está incluido
en `requirements.txt`. Un ejemplo para guardar otra corrida por separado es:

```bash
python -m papermill proyecto_inmuebles.ipynb resultados/analisis_p75.ipynb -p run_id departamentos_p75 -p cuantil 0.75
```

El plan B de la teoría es ejecutar el notebook o el script con parámetros
explícitos. Para otra corrida local: `python ejecutar.py --run-id departamentos_p80
--cuantil 0.80` (en una sola línea). Cambiar el cuantil cambia el problema:
sus métricas no sirven para decidir retrospectivamente qué etiqueta conviene.

## Fuente y unidad de análisis

Archivo original: `datasets/departamentos_2020.csv`, BA Data, muestra de avisos
de departamentos en venta durante 2020. Fuente:
https://data.buenosaires.gob.ar/dataset/departamentos-venta

Descarga directa verificada:
https://cdn.buenosaires.gob.ar/datosabiertos/datasets/secretaria-de-desarrollo-urbano/departamentos-venta/departamentos-en-venta-2020.csv

Fecha de descarga, SHA-256 y perfil original: `datasets/fuentes_y_perfil.json`.
Los otros dos datasets descargados se conservan como comparación; no se mezclan
con el entrenamiento. Una fila es una publicación, no una compraventa efectiva
ni una propiedad única. No incluye antigüedad. Se conservan las zonas de la fuente
(por ejemplo, Flores Norte/Sur) y se normaliza la escritura de los nombres.

## Decisiones metodológicas

1. Se eliminan duplicados exactos y precios ausentes/no positivos. Superficies
   fuera de 10–1000 m² y ambientes fuera de 1–20 se marcan como faltantes:
   son límites de plausibilidad declarados, no una corrección certificada del dato.
   Se unifican espacios, tildes y el error de escritura de Núñez.
2. Se agrupa por dirección normalizada. Un 20 % de los grupos queda en prueba;
   el porcentaje de filas puede diferir. Una dirección puede contener varias
   unidades y agruparlas es conservador. Distintas escrituras aún pueden dejar
   publicaciones de la misma propiedad en grupos diferentes.
3. El umbral es el percentil 75 del precio **sólo en entrenamiento**. Clase 1:
   precio mayor al umbral; clase 0: precio menor o igual. Es una definición
   relativa de segmento alto, no una clasificación objetiva de lujo.
4. Predictores: superficie, ambientes y barrio. No entran precio en dólares,
   pesos, precio/m², cotización, dirección ni clase. El precio/m² tampoco se
   utiliza para reconstruir superficies, porque incorporaría el objetivo.
5. Medianas, escalas y categorías se aprenden dentro del `Pipeline` en cada fold.
   Se usa `StratifiedGroupKFold` de tres folds, con las mismas particiones para
   todos los modelos. La estratificación es aproximada por la restricción de grupos.
6. Se compara `DummyClassifier`, Regresión Logística y Random Forest.
   `GridSearchCV` busca C y ponderación de clases para logística; profundidad
   y mínimo de muestras por hoja para el bosque. Se elige entre los dos modelos
   predictivos por F1 medio de clase alta en CV, antes de evaluar prueba.
7. Se publican Accuracy, Precision, Recall, F1 y matrices de confusión de todos
   los modelos. La clase positiva es alto. La evaluación final usa umbral de
   probabilidad 0.5; no se optimiza contra prueba. Las desviaciones entre folds
   no son intervalos de confianza y la CV usada para selección no es una
   estimación independiente del rendimiento final.

## Entregables generados

| Archivo o carpeta | Contenido |
| --- | --- |
| `resultados/analisis_ejecutado.ipynb` | Notebook con resultados y gráficos |
| `presentacion/defensa.pdf` | Siete filminas con resultados reales |
| `resultados/departamentos_2020/proyecto.sqlite` | Base accesible sin servidor |
| `metricas.csv`, `resumen.json` | Comparación real, auditoría y configuración |
| `predicciones.csv` | Predicción y probabilidad en prueba, por modelo |
| `busqueda_*.csv` | Resultados de cada combinación y fold |
| `modelo.joblib` | Pipeline elegido, umbral de precio y configuración |
| `graficos/` | Exploración, clases, comparación, confusión y variables |
| `configuracion.json`, `ejecucion.log` | Parámetros, versiones y trazabilidad |

Las rutas de las filas cortas son relativas a `resultados/departamentos_2020/`.
Repetir un `run_id` reemplaza sus resultados; usar otro para conservar ambas
corridas. El modelo guardado se ajusta sólo en entrenamiento para mantener
su correspondencia con las métricas reportadas. Cargar únicamente archivos
joblib de confianza; no se requiere deserializarlos para consultar las métricas.

## Base SQLite

`datos_limpios` guarda predictores limpios, precio para auditoría, etiqueta y
partición. `datos_preprocesados` guarda la representación numérica efectivamente
usada por el modelo elegido, transformada con su preprocesador ajustado en train.
`predicciones` enlaza `fila_fuente` con valores reales, predichos y probabilidades.
`metricas` guarda CV y prueba; `configuracion` guarda los parámetros de la corrida.
Los índices únicos permiten detectar duplicación de filas o predicciones.

Se puede abrir la BD con DB Browser for SQLite o consultar con Python:

```python
import sqlite3
import pandas as pd

with sqlite3.connect("resultados/departamentos_2020/proyecto.sqlite") as conexion:
    print(pd.read_sql_query("SELECT * FROM metricas", conexion))
```

## Relación con la consigna y la teoría

| Requisito | Implementación |
| --- | --- |
| Código modular, funciones y clases | Paquete `inmuebles`, dataclass de configuración |
| Pipeline con imputación, escala y categóricas | `inmuebles/modelos.py` |
| Dos clasificadores y búsqueda de hiperparámetros | Logística + bosque; GridSearchCV |
| Accuracy, Precision, Recall, F1 y matriz | CSV, JSON, notebook y gráficos |
| Datos de entrada y resultados en BD | Cinco tablas SQLite |
| Visualizaciones | Cinco figuras reproducibles |
| Entorno reproducible y logging | Dependencias fijadas y log por corrida |
| Notebook reproducible | Celda `parameters`, ejecución parametrizada con Papermill |
| Presentación de 6–8 filminas | Presentación PDF local de siete páginas |
| Git/GitHub y entrega al campus | Pendiente por decisión del usuario; sin operaciones Git |

La ejecución automatizada se realiza con Papermill. Las dependencias fijadas reflejan las versiones usadas;
no equivalen a certificar su instalación desde cero en todos los sistemas.

## Verificación

`python -m unittest discover -s tests -v` comprueba separación por dirección,
independencia del umbral respecto a prueba, admisión de barrios nuevos,
ausencia de precios en los predictores y persistencia SQLite repetible.
La verificación completa consiste además en ejecutar el notebook desde un kernel
nuevo y revisar los archivos de salida y las filas guardadas.

## Límites que deben explicarse en la defensa

Los datos describen precios publicados en 2020 y no validan tasaciones actuales.
La muestra está muy concentrada en el segundo trimestre y no representa un censo
equilibrado de CABA. Hay pocos predictores, errores de superficie y publicaciones
repetidas. La división por dirección reduce la fuga, pero no identifica cada
propiedad de forma infalible. Un precio alto puede reflejar tamaño o ubicación;
no demuestra calidad, lujo, causalidad ni el precio de cierre de la operación.
La ejecución guardada eligió Regresión Logística por F1 medio en CV. Sus métricas
de prueba son Accuracy 0,8953, Precision 0,7425, Recall 0,8878 y F1 0,8087.
El umbral de precio calculado fue USD 235.000. No se reajustaron decisiones con
estas métricas de prueba. La base mayoritaria logra Accuracy 0,7508 y F1 0,
por lo que accuracy por sí sola daría una impresión incompleta del rendimiento.
