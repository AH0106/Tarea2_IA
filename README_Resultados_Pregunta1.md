# Pregunta 1 · Resultados con el dataset de Spotify

En la rama `main2` se evaluaron dos tareas sobre las mismas canciones: predecir la popularidad (`popularity`) y clasificar el contenido explícito (`explicit`). En ambas tareas, las versiones de **10 y 5 variables** obtuvieron los mejores resultados en validación y pasaron a prueba. Las cinco variables seleccionadas dependen del objetivo.

Este README registra cómo se obtuvieron las tablas del notebook en el commit `99cad99`. [Notebook ejecutado](notebooks/tarea2_p1.ipynb)

## Datos y particiones

El CSV de Spotify contiene 114.000 filas y 20 columnas. Se conservó la primera aparición de cada `track_id`, lo que dejó 89.741 canciones. Después se excluyeron las filas con valor cero en al menos una de estas columnas: `tempo`, `duration_ms`, `danceability` o `popularity`. Las métricas corresponden a las **80.140 canciones** que pasaron esos filtros. [CSV utilizado](notebooks/data/raw/spotify-tracks-dataset-detailed.csv)

| Partición | Canciones | Uso |
|---|---:|---|
| Entrenamiento (70 %) | 56.098 | Calcular los rankings y ajustar el escalado y los modelos |
| Validación (15 %) | 12.021 | Comparar las tres versiones de cada tarea y elegir dos |
| Prueba (15 %) | 12.021 | Evaluar las dos versiones elegidas de cada tarea, con el ajuste de entrenamiento |

La división usa la semilla `42`. Cada tarea parte de diez variables y usa su propio ranking de entrenamiento para definir los subconjuntos de cinco y tres. La tabla muestra ambos rankings de mayor a menor. Los puntajes de las dos columnas miden cantidades diferentes.

| Posición | Variable para popularidad | Correlación absoluta de Pearson | Variable para contenido explícito | Información mutua |
|---:|---|---:|---|---:|
| 1 | `instrumentalness` | 0,193335 | `speechiness` | 0,046522 |
| 2 | `speechiness` | 0,087724 | `instrumentalness` | 0,012128 |
| 3 | `danceability` | 0,076011 | `duration_ms` | 0,011801 |
| 4 | `loudness` | 0,075518 | `loudness` | 0,010403 |
| 5 | `duration_ms` | 0,062281 | `energy` | 0,009634 |
| 6 | `liveness` | 0,051351 | `tempo` | 0,009247 |
| 7 | `energy` | 0,034354 | `acousticness` | 0,008883 |
| 8 | `tempo` | 0,018312 | `danceability` | 0,006814 |
| 9 | `acousticness` | 0,004791 | `liveness` | 0,004116 |
| 10 | `valence` | 0,002733 | `valence` | 0,003074 |

## 1.A · Predicción de popularidad

Cada versión combina `StandardScaler` y `LinearRegression`. El criterio de selección fue el **mayor R² en validación**. RMSE y MAE expresan el error en puntos de popularidad. Como referencia, predecir para cada canción la popularidad media de entrenamiento dio un RMSE de **18,1680** en validación.

| Partición | Variables | R² | RMSE | MAE |
|---|---:|---:|---:|---:|
| Entrenamiento | 10 | 0,069231 | 17,500183 | 14,375249 |
| Entrenamiento | 5 | 0,054217 | 17,640760 | 14,600123 |
| Entrenamiento | 3 | 0,051832 | 17,662994 | 14,610081 |
| Validación | 10 | 0,063661 | 17,580138 | 14,457106 |
| Validación | 5 | 0,052301 | 17,686465 | 14,670524 |
| Validación | 3 | 0,051261 | 17,696167 | 14,682621 |
| Prueba | 10 | 0,074063 | 17,405252 | 14,292572 |
| Prueba | 5 | 0,057681 | 17,558546 | 14,550657 |

## 1.B · Clasificación de contenido explícito

El equipo eligió **información mutua** para ordenar las variables y **ROC-AUC de validación** para seleccionar los modelos. El notebook calcula el ranking con `mutual_info_classif` sobre entrenamiento y con semilla `42`. Después ajusta una regresión logística por subconjunto, con `StandardScaler` y `max_iter=1000`.

ROC-AUC mide la capacidad del modelo para asignar una probabilidad más alta a una canción explícita que a una canción sin contenido explícito. Se calculó con las probabilidades de la clase 1. Las otras cuatro métricas utilizan las clases predichas con el umbral predeterminado. Precisión, recall y F1 se refieren a la clase explícita.

| Partición | Variables | ROC-AUC | Exactitud | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Entrenamiento | 10 | 0,818150 | 0,913758 | 0,522037 | 0,109154 | 0,180556 |
| Entrenamiento | 5 | 0,789550 | 0,914489 | 0,556728 | 0,086422 | 0,149619 |
| Entrenamiento | 3 | 0,782999 | 0,914667 | 0,565934 | 0,084374 | 0,146854 |
| Validación | 10 | 0,821495 | 0,916479 | 0,560000 | 0,108949 | 0,182410 |
| Validación | 5 | 0,799291 | 0,916729 | 0,595745 | 0,081712 | 0,143713 |
| Validación | 3 | 0,789574 | 0,916064 | 0,571429 | 0,073930 | 0,130922 |
| Prueba | 10 | 0,799335 | 0,917311 | 0,566038 | 0,117417 | 0,194489 |
| Prueba | 5 | 0,772634 | 0,917894 | 0,607362 | 0,096869 | 0,167089 |

### Matriz de confusión en validación · 10 variables

En validación, 1.028 de las 12.021 canciones tenían contenido explícito (8,55 %). Asignar a todas la clase mayoritaria habría dado una exactitud de 91,45 %. De las 1.028 canciones explícitas, el modelo clasificó 112 como explícitas.

| Clase real | Predicción sin contenido explícito | Predicción con contenido explícito |
|---|---:|---:|
| Sin contenido explícito | 10.905 | 88 |
| Con contenido explícito | 916 | 112 |

## Reproducir los resultados

Desde la raíz de una copia del repositorio en `main2`, con Python 3.12 y una terminal de macOS o Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab notebooks/tarea2_p1.ipynb
```

En JupyterLab, elegir el kernel del entorno y ejecutar **Kernel → Restart Kernel and Run All Cells**. El notebook lee el CSV incluido en el repositorio. En la verificación del commit `99cad99`, las 19 celdas de código produjeron salidas idénticas en dos ejecuciones desde kernels limpios. El archivo de dependencias especifica versiones mínimas. [Dependencias del proyecto](requirements.txt)

## Pendiente del equipo

Queda por redactar en el notebook la justificación de las variables y de la métrica elegida, interpretar las tablas de 1.A y 1.B y completar el análisis de generalización de 1.C. La integración de esta copia con la Pregunta 2 corresponde al paso final de coordinación.
