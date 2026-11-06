# Pregunta 2 · Plan de trabajo

El objetivo es comparar cuánto cambia el desempeño de regresión logística y SVM al incorporar pseudoetiquetas, con el 10 % y el 30 % de las etiquetas de entrenamiento conocidas. El trabajo se desarrolla en la Parte 2 del [notebook de la tarea](notebooks/tarea2.ipynb).

## 1. Revisar el dataset y justificar las características

**Estado actual.** El [CSV local de SDSS17](notebooks/data/raw/star_classification.csv) contiene 100.000 observaciones. La columna objetivo `class` distingue GALAXY (59.445), STAR (21.594) y QSO (18.961). El notebook selecciona siete columnas de observación e instrumentación. Una de ellas, `rerun_ID`, vale 301 en todas las filas.

**Trabajo del equipo.** Revisar valores faltantes, duplicados y columnas constantes. Distinguir las mediciones de los identificadores y justificar qué información aporta cada predictor elegido.

- [ ] La sección 2.0 documenta al menos 15.000 filas utilizables, siete características utilizables y el número de observaciones por clase después de la limpieza.

## 2. Verificar la partición y las etiquetas conocidas

**Estado actual.** El notebook separa entrenamiento y prueba en proporción 80/20 con estratificación. En `crear_escenarios`, la fracción solicitada corresponde al **segundo** conjunto de índices devuelto. La celda que imprime los conteos usa ese conjunto como conocido, pero la regresión logística usa el primero. Así, entrena con el 90 % en el escenario del 10 % y con el 70 % en el del 30 %.

**Trabajo del equipo.** Corregir la asignación y comprobar los índices que recibe realmente cada modelo. Si se conservan las 100.000 filas, se reservan 20.000 para prueba y los conteos de entrenamiento deben ser los siguientes.

| Escenario | Conocidas | Ocultas | Total de entrenamiento |
|---|---:|---:|---:|
| 10 % | 8.000 | 72.000 | 80.000 |
| 30 % | 24.000 | 56.000 | 80.000 |

- [ ] Para cada semilla y escenario, ambos grupos de índices son disjuntos y juntos cubren el entrenamiento. Si la limpieza cambia el total, se recalculan los conteos. Todos los enfoques comparados usan la misma partición y las mismas etiquetas conocidas para ese par de semilla y escenario.

## 3. Establecer la referencia supervisada

**Estado actual.** La sección 2.A ensaya tres valores de `C` para regresión logística con cinco pliegues y F1 macro. Usa la semilla 42 y está pendiente la corrección de índices de la etapa 2. La sección de SVM está por desarrollar.

**Trabajo del equipo.** En cada escenario, justificar y evaluar al menos tres configuraciones por clasificador mediante validación cruzada sobre las muestras con etiqueta conocida. Registrar la función de pérdida de cada modelo y distinguirla de la métrica empleada para elegir la configuración.

- [ ] Hay un registro por ensayo con escenario, semilla, clasificador, hiperparámetros, pérdida utilizada y métricas de validación. Los puntajes de validación cruzada se identifican como `cv` (la celda actual los guarda como `train`). La selección utiliza exclusivamente los datos de entrenamiento.

## 4. Incorporar pseudoetiquetas

**Estado actual.** Las secciones 2.B y 2.C tienen apartados para agrupamiento, asignación de pseudoetiquetas y reentrenamiento. Falta realizar los ensayos.

**Trabajo del equipo.** Evaluar al menos dos configuraciones de K-Means y dos de DBSCAN. Definir y justificar cómo se asignará una clase a las muestras seleccionadas. La meta del 60 % cuenta **etiquetas reales y estimadas juntas**. Con 80.000 filas de entrenamiento, la meta es de 48.000 filas etiquetadas en total. Faltarían 40.000 pseudoetiquetas en el escenario del 10 % y 24.000 en el del 30 %.

- [ ] Cada ensayo de agrupamiento registra sus parámetros, las cantidades de etiquetas reales y estimadas, la cobertura lograda y la razón de cualquier diferencia respecto de la meta.
- [ ] Se evalúan las cuatro combinaciones de agrupamiento y clasificador con al menos tres configuraciones del clasificador por combinación. En cada pliegue, las etiquetas de validación se reservan para puntuar. Las etiquetas ocultas y el conjunto de prueba quedan fuera del pseudoetiquetado y de la selección.

## 5. Comparar los seis enfoques y redactar el análisis

**Estado actual.** El notebook declara las semillas 42, 7 y 123, pero el ensayo de regresión logística usa solo 42. El [registro CSV actual](notebooks/resultados.csv) contiene una fila de ejemplo con F1 macro y exactitud iguales a cero.

**Trabajo del equipo.** Una vez elegidas las configuraciones por validación, evaluar en prueba los dos modelos supervisados y los cuatro modelos con pseudoetiquetas. Comparar cada clasificador con su referencia supervisada bajo el mismo escenario. Interpretar cómo influyen los hiperparámetros, la cantidad de etiquetas reales y la calidad de las pseudoetiquetas, citando las mediciones que sustentan cada conclusión.

- [ ] La tabla final contiene **36 evaluaciones de prueba** (6 enfoques × 2 escenarios × 3 semillas), una por combinación. El registro conserva por separado los ensayos de validación y excluye la fila de ejemplo. El equipo resume los resultados entre semillas y verifica que el notebook se ejecute de principio a fin.

**Uso de asistencia generativa.** Se utilizó Codex para organizar los requisitos del enunciado y contrastarlos con el estado del repositorio. El equipo es responsable de diseñar y justificar la estrategia experimental y de elaborar el análisis de resultados.
