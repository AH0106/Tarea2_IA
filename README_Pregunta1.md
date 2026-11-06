# Pregunta 1 — Guía de trabajo

**Tarea 2 · CIT-2013 Inteligencia Artificial · UDP**

Documento de traspaso para completar la Pregunta 1 de forma autónoma.

---

## 0. Qué es y qué no es este documento

**Qué es:** el método completo para cerrar los apartados pendientes de la Pregunta 1.
Para cada uno: qué hay que hacer, por qué, el pseudocódigo, las funciones de las
librerías con sus argumentos, la salida esperada y cómo interpretarla.

**Qué no es:** las celdas resueltas. El código lo escribes tú.

Esto no es una formalidad. Hay dos razones concretas:

1. **Interrogación oral al azar.** El enunciado dice que algunos trabajos se
   seleccionan y que *la calificación puede modificarse según el desempeño en esa
   instancia*. Un apartado que no puedas explicar en voz alta puede bajarte la nota
   aunque el código esté correcto.
2. **El enunciado prohíbe** usar herramientas generativas «para el diseño de
   soluciones o la elaboración de análisis de resultados». Las decisiones y los
   textos de análisis tienen que ser de ustedes.

Lo que sí es legítimo y está aquí: el método, las funciones, y las trampas que
producen errores silenciosos.

---

## 1. Reglas de coordinación (leer antes de tocar nada)

### 1.1 El riesgo del archivo compartido

Los notebooks `.ipynb` son JSON con los outputs embebidos. Si dos personas editan
el mismo archivo y hacen push, Git no puede fusionarlo y el conflicto **puede
corromper el notebook entero**.

**Protocolo obligatorio:**

```
1. Haz una copia del notebook actual y renómbrala:   tarea2_p1.ipynb
2. Trabaja SOLO en ese archivo. No toques tarea2.ipynb.
3. Borra de tu copia todas las celdas de la Pregunta 2 (si hay alguna).
4. Haz commit y push solo de tarea2_p1.ipynb.
5. El viernes por la tarde se unifican: se copian las celdas de las dos partes
   a un único notebook final y se ejecuta completo de principio a fin.
```

El paso 5 no es opcional. La unificación tiene que terminar con un
**Kernel → Restart Kernel and Run All Cells** sin un solo error, porque el
enunciado dice que si los códigos no se ejecutan sin cambios la tarea se evalúa
con **nota mínima de inmediato**.

### 1.2 Lo que NO se cambia

El notebook ya tiene decisiones tomadas y celdas que funcionan. Cambiarlas rompe
la coherencia interna del informe y obliga a reescribir los apartados anteriores.

**No modificar:**

| Objeto | Por qué es intocable |
|---|---|
| La cadena de limpieza | el markdown del 1.2 ya reporta los conteos de filas; si cambias el filtrado, esos números quedan mentirosos |
| `seed = 42` | cambiar la semilla cambia la división y por lo tanto **todas** las métricas ya reportadas |
| La división 70/15/15 (`df_train`, `df_val`, `df_test`) | lo mismo |
| La lista de 10 predictoras | el ranking del 1.A.1 y los tres modelos del 1.A.2 dependen de ella |
| Las sublistas de 5 y de 3 variables del 1.A.1 | ya están justificadas en el markdown |
| Los tres modelos lineales entrenados | las métricas del 1.A.2 ya están reportadas |

**Antes de escribir una línea**, abre el notebook y anota los nombres exactos de
estos objetos, porque los vas a usar constantemente:

```python
# Ejecuta esto y copia los resultados a un papel o a un markdown de trabajo
print("predictoras:", predictoras)          # la lista de 10
print("sublista de 5:", listFive)           # verifica el nombre real en el notebook
print("sublista de 3:", listThree)
print("objetivo continuo Y1:", 'popularity')
print("objetivo binario Y2:", 'explicit')   # verifica el nombre real
print("tamaños:", len(df_train), len(df_val), len(df_test))
print(tabla)                                 # la tabla de métricas del 1.A.2
```

Si algún nombre no coincide, usa el del notebook. **No renombres nada.**

---

## 2. Estado y pendientes

### 2.1 Lo que ya está hecho

| Apartado | Contenido |
|---|---|
| **1.1** | presentación del dataset, origen, descripción de las 12 columnas, identificación de Y₁ e Y₂ |
| **1.2** | limpieza justificada con conteos de filas, división 70/15/15 verificada |
| **1.A.1** | ranking de las 10 predictoras por correlación con Y₁, y las sublistas de 5 y 3 |
| **1.A.2** | tres regresiones lineales entrenadas (10, 5 y 3 variables) con métricas en entrenamiento y validación |

### 2.2 Lo que falta

| Apartado | Qué es | Tiempo estimado |
|---|---|---|
| **1.A.3** | comparar las tres versiones en entrenamiento y validación | 1 h |
| **1.A.4** | evaluar las dos mejores en el conjunto de prueba | 30 min |
| **1.B.1** | seleccionar variables para Y₂ (ranking nuevo) | 1 h |
| **1.B.2** | entrenar tres regresiones logísticas | 1 h |
| **1.B.3** | métricas de clasificación e interpretación | 1,5 h |
| **1.B.4** | evaluar las dos mejores en prueba | 30 min |
| **1.C** | análisis final de la Pregunta 1 | 2 h |

**Total ≈ 7,5 h.** Prioridad si el tiempo aprieta: 1.A.3 y 1.A.4 primero (son los
más baratos, los modelos ya están entrenados), después 1.B.1 a 1.B.4, y 1.C al
final. **Ningún apartado se deja en blanco**: un apartado delgado puntúa parcial,
uno vacío puntúa cero.

---

## 3. Teoría mínima para escribir los análisis

No es relleno. Cada bloque corresponde a una afirmación que los apartados 1.A.3 y
1.C tienen que sostener.

### 3.1 Qué hace la regresión lineal

Supone que la respuesta es una suma ponderada de las predictoras:

```
ŷ  =  θ₀  +  θ₁·x₁  +  θ₂·x₂  +  ...  +  θₙ·xₙ
```

| Símbolo | Qué es | En sklearn |
|---|---|---|
| `ŷ` | la predicción | lo que devuelve `.predict()` |
| `θ₀` | intercepto: el valor predicho cuando todas las x valen 0 | `.intercept_` |
| `θᵢ` | coeficiente: cuánto cambia ŷ si xᵢ sube 1 unidad y el resto no se mueve | `.coef_` |

Las diapositivas del ramo usan **θ**; el libro de Russell & Norvig usa **w**. Es lo
mismo con otra letra.

### 3.2 Qué minimiza

El error cuadrático medio, que es la función de costo del formulario del ramo:

```
J(θ)  =  (1 / 2m) · Σ ( h_θ(x⁽ⁱ⁾) − y⁽ⁱ⁾ )²
```

donde `m` es el número de observaciones, `h_θ(x⁽ⁱ⁾)` la predicción para la
observación *i*, e `y⁽ⁱ⁾` su valor real.

El error se **eleva al cuadrado** por dos razones: los errores positivos y
negativos no se cancelan entre sí, y los errores grandes pesan mucho más que los
pequeños. Esa segunda consecuencia explica por qué el RMSE siempre es mayor o
igual que el MAE.

### 3.3 Cómo encuentra el mínimo (dato que suelen equivocar)

Para la regresión lineal el mínimo se calcula **directamente con una fórmula**, sin
iterar. Es la **ecuación normal**:

```
θ*  =  (XᵀX)⁻¹ Xᵀy
```

Esa es la ruta que usa `LinearRegression` de sklearn. **No usa gradiente
descendiente.** Afirmar lo contrario en el análisis es un error.

El gradiente descendiente —la regla `θᵢ ← θᵢ − α · ∂J/∂θᵢ` del formulario— es el
método iterativo que se usa cuando la fórmula directa no existe, y es lo que sí
usa `LogisticRegression`.

**Consecuencia práctica:** como la fórmula directa encuentra el mínimo exacto,
estandarizar las variables **no cambia** el R² ni el RMSE de una regresión lineal
sin regularización. Solo cambia la escala de los coeficientes, y con eso su
interpretabilidad. En la logística, la SVM y cualquier modelo regularizado, en
cambio, escalar **sí cambia los resultados**.

### 3.4 Overfitting y underfitting

El diagnóstico sale de comparar entrenamiento con validación.

| Diagnóstico | Señal | Qué está pasando |
|---|---|---|
| **Overfitting** | R² alto en entrenamiento, bastante más bajo en validación: **brecha grande** | el modelo memorizó el ruido del entrenamiento |
| **Underfitting** | R² bajo en **ambos**, con brecha casi nula | el modelo es demasiado simple, o las predictoras no contienen la señal |
| **Buen ajuste** | R² alto en ambos, brecha pequeña | el caso deseable |

**Por qué un R² bajo suele ser underfitting y no un bug:** el R² alcanzable por una
regresión lineal está acotado por la correlación entre predictoras y objetivo. Con
una sola predictora se cumple exactamente **R² = r²**. Así que si la correlación
más alta de la matriz es 0,30, ninguna regresión lineal con esas variables va a
pasar de un R² del orden de 0,09 por esa vía. **El techo está en los datos, no en
el código**, y se puede saber antes de entrenar.

### 3.5 Las tres métricas, y cómo verificarlas

| Métrica | Qué mide | Unidades | Mejor cuando |
|---|---|---|---|
| **R²** | qué fracción de la varianza del objetivo explica el modelo | ninguna (0 a 1) | es mayor |
| **RMSE** | error típico, castigando los errores grandes | las del objetivo | es menor |
| **MAE** | error promedio en valor absoluto | las del objetivo | es menor |

Están ligadas por una identidad, y eso sirve para verificar:

```
R²  =  1  −  ( RMSE / σ )²        σ = desviación estándar del objetivo
```

Dos lecturas:

- **R² = 0** equivale a **RMSE = σ**: el modelo no mejora respecto a predecir
  siempre la media. Ese es el punto de comparación honesto.
- Si los números no cumplen la identidad, **alguna métrica está mal calculada**.

Además **RMSE ≥ MAE siempre**. Si el cociente RMSE/MAE ronda 1,25, los errores se
distribuyen de forma aproximadamente normal; mucho mayor indica unos pocos errores
muy grandes.

### 3.6 Multicolinealidad

Ocurre cuando dos predictoras están fuertemente correlacionadas **entre sí**. El
modelo no puede distinguir el efecto de una del de la otra y reparte el peso
arbitrariamente: los coeficientes se vuelven grandes, inestables y a veces con el
signo contrario al esperado.

No arruina las predicciones, pero **arruina la interpretación de los
coeficientes**. Se detecta en la matriz de correlaciones entre predictoras:
valores por encima de 0,7 en valor absoluto son señal de alerta.

---

## 4. Apartado 1.A.3 — Comparar en entrenamiento y validación

**Qué exige el enunciado:** «comparar el desempeño de las tres versiones en los
conjuntos de entrenamiento y validación, utilizando métricas apropiadas».

Una tabla no es una comparación: es materia prima. La comparación es el texto.

### 4.1 Primero: verificar que las métricas son las que parecen

Antes de escribir nada. Si el «RMSE» de la tabla fuera en realidad un MSE,
escribirías un análisis entero sobre números equivocados.

```
sigma ← desviación estándar de df_val[objetivo_y1]
imprimir "sigma (RMSE del modelo trivial):", sigma

PARA CADA fila EN tabla:
    r2_reconstruido ← 1 − (fila.RMSE_val / sigma) ** 2
    imprimir fila.version, fila.R2_val, r2_reconstruido     # deben coincidir
```

**Funciones:**

```
df[col].std()            → desviación estándar
tabla.itertuples()       → recorre las filas de un DataFrame; cada fila
                           permite acceder a las columnas con un punto:
                           fila.R2_val, fila.version
** 2                     → elevar al cuadrado
```

Si no coinciden, para y arregla las métricas antes de seguir.

### 4.2 Cálculos de apoyo

```
# Brechas de generalización
PARA CADA fila EN tabla:
    imprimir fila.version, "brecha:", fila.R2_train − fila.R2_val

# Caídas al quitar variables, EN VALIDACIÓN
caida_10_5 ← R2_val del completo  − R2_val del reducido
caida_5_3  ← R2_val del reducido  − R2_val del minimo

# Mejora sobre el modelo trivial
PARA CADA fila EN tabla:
    imprimir fila.version, "mejora sobre la media:", (sigma − fila.RMSE_val) / sigma

# Cociente RMSE/MAE
PARA CADA fila EN tabla:
    imprimir fila.version, fila.RMSE_val / fila.MAE_val
```

### 4.3 El texto: tres párrafos

1. **Nivel absoluto.** Comparar el RMSE contra σ, que es lo que daría predecir
   siempre la media, y decir cuánto mejora el modelo sobre ese mínimo. Un RMSE casi
   igual a σ significa que el modelo aporta muy poco.

2. **Generalización.** Las tres brechas R²_train − R²_val y el diagnóstico que
   implican según la tabla de 3.4. Cuidado: brechas minúsculas con R² bajo en ambos
   conjuntos es **underfitting**, no un buen resultado. Es el error de lectura más
   común de este apartado.

3. **Efecto de quitar variables.** Comparar `caida_10_5` contra `caida_5_3`. Si
   quitar de 10 a 5 cuesta mucho más que de 5 a 3, el ranking univariado no estaba
   capturando bien la contribución conjunta de las variables, y eso merece
   comentario: una variable puede ser poco predictiva por sí sola y aportar mucho
   junto a otra.

### 4.4 Verificación de coherencia que delata errores

**R²_train debe ir de mayor a menor** al quitar variables. En entrenamiento,
agregar variables a una regresión por mínimos cuadrados nunca empeora el ajuste:
en el peor caso el modelo les asigna coeficiente cero. **Si el orden no se cumple
en entrenamiento, hay un error en el código.**

En **validación** esa garantía no existe. Si ahí el orden se rompe, es un resultado
legítimo e interesante de comentar, no un bug.

### 4.5 Gráfico que vale la pena

```
y_real ← df_val[objetivo_y1]
y_pred ← modelos['completo'].predict(df_val[predictoras])

plt.figure(figsize=(7, 6))
plt.scatter(y_real, y_pred, alpha=0.2, s=10)
# Diagonal de referencia: donde caerían los puntos si el modelo fuera perfecto
plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
plt.xlabel("Valor real")
plt.ylabel("Valor predicho")
plt.title("Real vs predicho (validación)")
plt.show()
```

**Funciones:**

```
plt.figure(figsize=(ancho, alto))   → figura nueva, tamaño en pulgadas
plt.scatter(x, y, alpha=, s=)       → nube de puntos. alpha es opacidad (0 a 1):
                                      bajo hace los puntos transparentes y las
                                      zonas oscuras revelan dónde se acumulan
plt.plot(x, y, 'r--')               → línea; 'r' roja, '--' punteada
plt.show()                          → dibuja
```

**Cómo interpretarlo:** si el modelo fuera perfecto, todos los puntos caerían sobre
la diagonal. Una nube **horizontal** —predicciones concentradas en un rango
estrecho mientras los valores reales se extienden por todo el eje— es la versión
visual del underfitting. Es la evidencia más convincente que se puede poner en 1.C.

**Usar validación, nunca prueba.**

---

## 5. Apartado 1.A.4 — Evaluar en prueba

**Qué exige:** evaluar las dos mejores versiones en el conjunto de prueba.

### 5.1 Por qué prueba se usa una sola vez

| Conjunto | Para qué sirve | Cuántas veces |
|---|---|---|
| Entrenamiento | el modelo ajusta sus coeficientes | todas |
| Validación | comparar versiones y elegir la mejor | muchas, es su función |
| **Prueba** | estimar el desempeño honesto del modelo ya elegido | **una sola vez** |

Si evalúas en prueba, cambias el modelo porque salió mal, y vuelves a evaluar,
entonces las decisiones ya están influidas por prueba: deja de ser una estimación
del desempeño con datos nuevos y se convierte en un segundo conjunto de validación.

### 5.2 Pseudocódigo

```
# Declarar la métrica que decide ANTES de mirar los resultados.
# Para regresión, R2 en validación es la elección natural.
dos_mejores ← las 2 versiones con mayor R2_val

y1_test ← df_test[objetivo_y1]
filas   ← lista vacía

PARA CADA nombre EN dos_mejores:
    pipe     ← modelos[nombre]        # YA entrenado: NO reentrenar
    columnas ← configs[nombre]

    pred ← pipe.predict( df_test[columnas] )

    agregar a filas:
        version   = nombre
        n_vars    = cantidad de columnas
        R2_test   = r2_score(y1_test, pred)
        RMSE_test = root_mean_squared_error(y1_test, pred)
        MAE_test  = mean_absolute_error(y1_test, pred)

tabla_test ← pd.DataFrame(filas)
imprimir tabla_test
```

**Funciones:**

```
pipe.predict(X)                          → array de predicciones, una por fila
r2_score(y_real, y_pred)                 → el R². EL ORDEN IMPORTA
root_mean_squared_error(y_real, y_pred)  → el RMSE
mean_absolute_error(y_real, y_pred)      → el MAE
pd.DataFrame(lista_de_diccionarios)      → una fila por diccionario
```

**Nota de versión:** `root_mean_squared_error` existe desde scikit-learn 1.4. Si el
import falla, usa `mean_squared_error(y_real, y_pred, squared=False)`. Para ver la
versión instalada: `import sklearn; print(sklearn.__version__)`.

### 5.3 Cómo interpretarlo

- **Prueba parecida a validación** → buena señal: la partición fue representativa y
  la estimación del desempeño es confiable.
- **Prueba bastante peor** → sospecha de sobreajuste a validación: se eligió una
  versión que funcionaba bien ahí por casualidad.
- **Prueba bastante mejor** → revisar la partición. Con conjuntos de miles de filas,
  una diferencia grande es más probablemente un error que suerte.

### 5.4 Lo que tiene que quedar escrito

**Qué métrica se usó para elegir las dos mejores.** Decir «las dos mejores» sin
indicar según qué criterio es un hueco que un corrector detecta y que en una
interrogación oral se pregunta de inmediato.

---

## 6. La regresión logística: qué cambia

Mismo dataset, mismas 10 predictoras, pero el objetivo ahora es **binario**: 0 ó 1.
Eso cambia el modelo, las métricas y los modos de fallar.

### 6.1 Por qué no sirve la regresión lineal aquí

Una regresión lineal predice cualquier número real: podría predecir −0,4 ó 1,7, que
no son valores posibles para una variable binaria. La logística lo resuelve en dos
movimientos:

1. Calcula una **combinación lineal** igual que antes: `z = θ₀ + θ₁x₁ + ... + θₙxₙ`
2. Pasa ese `z` por la **función sigmoide**, que lo comprime al intervalo (0, 1):

```
                    1
σ(z)  =  ─────────────────────
              1  +  e^(−z)

  z → −∞   ⟹  σ(z) → 0
  z =  0   ⟹  σ(z) = 0.5
  z → +∞   ⟹  σ(z) → 1
```

El resultado es **P(y = 1 | x)**: la probabilidad de que la observación pertenezca
a la clase 1. Para convertirla en decisión se aplica un **umbral**, por defecto 0,5.

### 6.2 La distinción que más errores causa

De lo anterior sale esto:

| Método | Devuelve | Para qué métricas |
|---|---|---|
| `predict(X)` | la **clase** ya decidida: 0 ó 1 | accuracy, precisión, recall, F1 |
| `predict_proba(X)` | las **probabilidades**: tabla de 2 columnas | **ROC-AUC** |

`predict_proba(X)[:, 1]` es la columna de P(clase = 1). La notación significa
«todas las filas, columna 1». La columna 0 es P(clase = 0), y las dos suman 1.

**Confundirlos no lanza error.** Ver la trampa de la sección 11.

### 6.3 Qué minimiza, y por qué sí usa gradiente descendiente

No minimiza el error cuadrático sino la **log-verosimilitud negativa**, también
llamada entropía cruzada:

```
J(θ)  =  −(1/m) · Σ [ y⁽ⁱ⁾ · log(h_θ(x⁽ⁱ⁾))  +  (1−y⁽ⁱ⁾) · log(1 − h_θ(x⁽ⁱ⁾)) ]
```

Conviene leerla por partes. Para una observación con `y = 1`, el segundo término se
anula y queda `−log(h)`: si el modelo le asignó probabilidad alta, el costo es
cercano a 0; si le asignó probabilidad cercana a 0, el costo se dispara. Para
`y = 0` ocurre lo simétrico. **La función castiga la confianza equivocada.**

A diferencia del caso lineal, esta función **no tiene solución cerrada**: no existe
una fórmula tipo ecuación normal que dé el mínimo. Por eso se resuelve iterando:

```
θᵢ  ←  θᵢ  −  α · ∂J/∂θᵢ        α = tasa de aprendizaje
```

De ahí viene el parámetro `max_iter`: el número máximo de iteraciones permitidas.

### 6.4 Por qué el escalado sí importa aquí

`LogisticRegression` de sklearn trae **regularización L2 activada por defecto**
(`penalty='l2'`), aunque no se escriba. La regularización añade al costo un término
que penaliza coeficientes grandes:

```
J_regularizado(θ)  =  J(θ)  +  λ · Σ θᵢ²
```

Esa penalización trata a todos los coeficientes **por igual**. Si una variable está
medida en cientos de miles, su coeficiente natural es diminuto y la penalización
casi no la afecta; si otra va de 0 a 1, su coeficiente es grande y resulta
castigada con fuerza. **El resultado depende de las unidades.**

Por eso aquí escalar cambia los números, no solo su interpretación. Son tres los
motivos para escalar siempre: la regularización, los métodos basados en distancia,
y la convergencia del gradiente descendiente.

---

## 7. Apartado 1.B.1 — Seleccionar variables para Y₂

**Qué haces:** un ranking **nuevo** de las mismas 10 variables, midiendo ahora su
relación con la variable binaria Y₂.

**Por qué:** el enunciado pide justificar la selección **para cada tarea**. El
ranking del 1.A.1 mide relación con Y₁ y no dice nada sobre Y₂. Reutilizarlo sin
más es un atajo que se nota. Además, la correlación de Pearson no es la herramienta
adecuada cuando el objetivo tiene solo dos valores: está diseñada para dos
variables continuas.

### 7.1 Opción A — diferencia de medias estandarizada

La idea: una variable sirve para distinguir dos clases si su **promedio es muy
distinto** en cada una.

```
y2_train ← df_train[objetivo_y2].astype(int)

# Promedio de cada predictora, separado por clase
medias ← df_train.groupby(objetivo_y2)[predictoras].mean()

puntajes ← diccionario vacío

PARA CADA variable EN predictoras:
    media_clase_1 ← medias.loc[1, variable]
    media_clase_0 ← medias.loc[0, variable]
    desviacion    ← df_train[variable].std()

    # Dividir por la desviación hace comparables escalas distintas
    puntajes[variable] ← |media_clase_1 − media_clase_0| / desviacion

ranking_y2 ← pd.Series(puntajes).sort_values(ascending=False)
imprimir ranking_y2
```

**Dividir por la desviación es obligatorio.** Sin esa división, una variable medida
en cientos de miles tendrá una diferencia de medias enorme en términos absolutos y
aparecerá primera, aunque discrimine peor que una que va de 0 a 1. Es el mismo
problema de escala que resuelve el `StandardScaler`.

### 7.2 Opción B — información mutua

Mide cuánta información sobre Y₂ aporta conocer la variable, sin suponer ninguna
forma concreta de relación.

```
from sklearn.feature_selection import mutual_info_classif

y2_train ← df_train[objetivo_y2].astype(int)

puntajes ← mutual_info_classif(df_train[predictoras], y2_train,
                               random_state=seed)

ranking_y2 ← pd.Series(puntajes, index=predictoras).sort_values(ascending=False)
```

**Funciones:**

```
df.groupby(columna)[lista].mean()   → promedio de cada columna de la lista,
                                      por cada valor de la columna agrupadora
medias.loc[1, 'variable']           → el valor de la fila etiquetada 1 y esa columna
df[col].std()                       → desviación estándar
abs(x)                              → valor absoluto
df[col].astype(int)                 → convierte True/False a 1/0
mutual_info_classif(X, y, random_state=)  → array de puntajes, uno por columna
pd.Series(valores, index=nombres)   → empareja cada valor con su nombre de columna
.sort_values(ascending=False)       → de mayor a menor
```

### 7.3 Qué se ve, y la verificación

Un ranking **distinto** al del 1.A.1. Si saliera idéntico en el mismo orden, lo más
probable es que se midió contra Y₁ por error.

Al final del apartado, definir dos listas nuevas **con nombres distintos** de las
del 1.A.1:

```
lista_5_y2 ← los 5 primeros de ranking_y2
lista_3_y2 ← los 3 primeros de ranking_y2

# Verificación de anidamiento: las 3 tienen que estar dentro de las 5
imprimir todas las de lista_3_y2 están en lista_5_y2
```

**No sobrescribir `listFive` ni `listThree`.** Si se reutilizan esos nombres, los
modelos del 1.A.2 quedan apuntando a las columnas equivocadas, y si alguna celda
anterior se vuelve a ejecutar, las métricas del 1.A cambian sin explicación
aparente.

### 7.4 El anidamiento, y por qué importa

Las sublistas tienen que ser **anidadas**: 10 ⊃ 5 ⊃ 3. Si no lo son, la comparación
entre versiones mezcla dos efectos —menos variables *y* variables distintas— y deja
de ser interpretable.

Si hay un **empate** o dos valores casi idénticos, declara la regla de desempate
*antes* de aplicarla, y que no use la misma cantidad que está empatada: eso es
circular. Criterios válidos: menor correlación con las otras predictoras ya
elegidas (aporta información nueva), o mayor pertinencia conceptual.

### 7.5 La justificación que hay que escribir

Dos niveles, los dos necesarios:

1. **Empírico** — el ranking, con los valores.
2. **Conceptual** — por qué tiene sentido en el dominio que esas variables predigan
   el objetivo. Un número sin interpretación no es una justificación.

---

## 8. Apartado 1.B.2 — Entrenar las tres logísticas

```
configs_y2 ← { 'completo': predictoras,
               'reducido': lista_5_y2,
               'minimo':   lista_3_y2 }

y2_train ← df_train[objetivo_y2].astype(int)
y2_val   ← df_val[objetivo_y2].astype(int)

modelos_y2    ← diccionario vacío
resultados_y2 ← lista vacía

PARA CADA (nombre, columnas) EN configs_y2.items():

    pipe ← make_pipeline( StandardScaler(),
                          LogisticRegression(max_iter=1000) )
    pipe.fit( df_train[columnas], y2_train )
    modelos_y2[nombre] ← pipe

    # CLASES: para accuracy, precisión, recall y F1
    pred_train ← pipe.predict( df_train[columnas] )
    pred_val   ← pipe.predict( df_val[columnas] )

    # PROBABILIDADES: para el ROC-AUC
    proba_val  ← pipe.predict_proba( df_val[columnas] )[:, 1]

    agregar a resultados_y2 las métricas de la sección 9
```

**Funciones:**

```
make_pipeline(paso1, paso2)      → encadena pasos; fit los ejecuta en orden
StandardScaler()                 → transforma a media 0 y desviación 1
LogisticRegression(max_iter=1000)
pipe.fit(X, y)                   → X son las PREDICTORAS, y el OBJETIVO
pipe.predict(X)                  → array de clases (0 ó 1)
pipe.predict_proba(X)            → tabla de 2 columnas con probabilidades
[:, 1]                           → todas las filas, columna 1
configs_y2.items()               → recorre clave Y valor del diccionario
```

**Falta `.items()`:** `for nombre, columnas in configs_y2:` falla con
`ValueError: too many values to unpack`. Sin `.items()`, Python entrega solo las
claves e intenta repartir cada texto en dos variables.

**Por qué `max_iter=1000`:** el valor por defecto es 100 y a menudo no alcanza.
Entonces sklearn emite `ConvergenceWarning`: el modelo **devuelve resultados
igualmente**, pero son los de un ajuste incompleto. Es una advertencia, no un
error, así que es fácil ignorarla. Si aparece, subir `max_iter`.

**Convertir el booleano a entero:** si Y₂ es de tipo `bool`, varias métricas
funcionan pero `confusion_matrix` etiqueta las filas de forma confusa. Conviene el
`.astype(int)` una sola vez y trabajar con 0/1 desde ahí.

---

## 9. Apartado 1.B.3 — Métricas de clasificación

```
# Dentro del bucle de la sección 8
agregar a resultados_y2:
    version        = nombre
    n_vars         = cantidad de columnas
    accuracy_train = accuracy_score(y2_train, pred_train)
    accuracy_val   = accuracy_score(y2_val,   pred_val)
    precision_val  = precision_score(y2_val, pred_val, zero_division=0)
    recall_val     = recall_score(y2_val,    pred_val, zero_division=0)
    f1_val         = f1_score(y2_val,        pred_val, zero_division=0)
    auc_val        = roc_auc_score(y2_val,   proba_val)       # PROBABILIDADES

# Fuera del bucle, solo para la mejor versión
mejor ← la versión con mayor f1_val
cols  ← configs_y2[mejor]
pred  ← modelos_y2[mejor].predict( df_val[cols] )

imprimir confusion_matrix(y2_val, pred)
imprimir classification_report(y2_val, pred)

# EL DATO QUE CONTEXTUALIZA TODO
imprimir y2_val.value_counts(normalize=True)
```

### 9.1 Qué mide cada métrica

| Métrica | Pregunta que responde | Fórmula | Cuándo engaña |
|---|---|---|---|
| **Accuracy** | ¿qué fracción del total acerté? | (VP+VN)/total | **con clases desbalanceadas** |
| **Precisión** | de las que marqué positivas, ¿cuántas lo eran? | VP/(VP+FP) | si casi nunca predice positivo |
| **Recall** | de las positivas reales, ¿cuántas detecté? | VP/(VP+FN) | si predice positivo casi siempre |
| **F1** | media armónica de precisión y recall | 2·P·R/(P+R) | equilibra las dos |
| **ROC-AUC** | ¿separa bien las clases en *todos* los umbrales? | área bajo la curva ROC | poco sensible al desbalance; 0,5 = azar |

El F1 usa la media **armónica**, que a diferencia de la aritmética es baja si
*cualquiera* de las dos es baja: no se puede sacar un F1 alto sacrificando una.

### 9.2 La matriz de confusión

```
                       PREDICHO
                   Neg(0)    Pos(1)
                 ┌─────────┬─────────┐
REAL   Neg(0)    │   VN    │   FP    │   ← falsos positivos: dijo 1, era 0
                 ├─────────┼─────────┤
       Pos(1)    │   FN    │   VP    │   ← falsos negativos: dijo 0, era 1
                 └─────────┴─────────┘

Precisión = VP / (VP + FP)      ← mira la COLUMNA de positivos
Recall    = VP / (VP + FN)      ← mira la FILA de positivos
```

**Filas contra columnas:** precisión y recall tienen el mismo numerador (VP) y
distinto denominador. La precisión divide por la **columna** (todo lo que predijo
positivo); el recall por la **fila** (todo lo que realmente era positivo). El
truco para no confundirlas: *precisión* evalúa las *predicciones*; *recall* evalúa
la cobertura de la *realidad*.

Si la columna «Pos» está casi vacía, el modelo casi nunca predice la clase
minoritaria.

### 9.3 El resultado previsible — leer esto antes de alarmarse

Si la clase positiva es, digamos, el 9 % de los datos, un modelo que responda
siempre «negativa» acierta el 91 % sin haber aprendido nada. Con un objetivo así,
lo habitual es obtener:

- **Accuracy alta** — parece excelente
- **Recall muy bajo, a veces exactamente 0** — casi no detecta la clase minoritaria
- **Precisión 0 o indefinida** — de ahí el `zero_division=0`
- **F1 muy bajo**
- **ROC-AUC moderado**, por encima de 0,5 — hay algo de señal, pero el umbral de
  0,5 no la aprovecha

**Esto no es un error de programación.** Es el comportamiento esperado, y
explicarlo *es* el análisis que se evalúa. Un F1 de 0,04 bien diagnosticado puntúa
más que un accuracy de 0,91 presentado como logro.

**Para qué sirve `zero_division=0`:** si el modelo nunca predice la clase positiva,
el denominador de la precisión es cero y la división es indefinida. Ese argumento
declara explícitamente que se quiere un 0 en ese caso, y el aviso desaparece.

### 9.4 Qué escribir en el 1.B.3

1. **La proporción de clases**, con el número concreto. Es el dato que contextualiza
   todo lo demás.
2. **Por qué el accuracy no sirve aquí**, calculando explícitamente qué accuracy
   tendría el modelo trivial que predice siempre la clase mayoritaria.
3. **Qué métrica se elige como principal y por qué.** Con desbalance, F1 o ROC-AUC.
4. **Qué revela la matriz de confusión** sobre el comportamiento concreto del
   modelo.

---

## 10. Apartado 1.B.4 — Evaluar en prueba

```
# Declarar la métrica decisora ANTES de mirar resultados.
# Con clases desbalanceadas: f1_val o auc_val, NO accuracy.
metrica_decisora ← 'f1_val'
dos_mejores ← las 2 versiones con mayor metrica_decisora en validación

y2_test ← df_test[objetivo_y2].astype(int)
filas   ← lista vacía

PARA CADA nombre EN dos_mejores:
    pipe     ← modelos_y2[nombre]
    columnas ← configs_y2[nombre]

    pred  ← pipe.predict( df_test[columnas] )
    proba ← pipe.predict_proba( df_test[columnas] )[:, 1]

    agregar a filas:
        version        = nombre
        accuracy_test  = accuracy_score(y2_test, pred)
        precision_test = precision_score(y2_test, pred, zero_division=0)
        recall_test    = recall_score(y2_test, pred, zero_division=0)
        f1_test        = f1_score(y2_test, pred, zero_division=0)
        auc_test       = roc_auc_score(y2_test, proba)

imprimir pd.DataFrame(filas)
```

**Coherencia interna:** con desbalance, elegir por accuracy sería incoherente con
lo que se acaba de argumentar en el 1.B.3. Esa contradicción interna es lo primero
que se nota al corregir.

---

## 11. Trampas: errores que NO lanzan excepción

Estos producen un notebook que corre perfecto y entrega números equivocados. Cada
uno tiene una verificación que lo detecta; conviene dejarla escrita en el notebook.

| Error | Síntoma | Verificación que lo detecta |
|---|---|---|
| **`predict` en vez de `predict_proba` para el ROC-AUC** | AUC bajo e incoherente con el F1 | usar `predict_proba(X)[:, 1]` |
| **MSE etiquetado como RMSE** | el «RMSE» es enorme frente al rango del objetivo | `R² = 1 − (RMSE/σ)²` |
| **No dividir por la desviación en 1.B.1** | el ranking lo domina la variable de escala más grande | que el ranking no esté ordenado por magnitud de la variable |
| **Ranking ordenado por valor con signo** | las correlaciones negativas fuertes caen al final | mirar el ranking impreso: arriba el mayor valor absoluto |
| **Ranking ordenado ascendente** | las mejores variables quedan al final | lo mismo |
| **Sublistas no anidadas** | la comparación entre versiones no es interpretable | comprobar que las 3 están dentro de las 5 |
| **Sobrescribir `listFive`/`listThree`** | las métricas del 1.A cambian solas | nombres distintos para las listas de Y₂ |
| **El objetivo entre las predictoras** | R² o F1 ≈ 1,0 | `objetivo in predictoras` debe dar `False` |
| **Un `fit` de prueba olvidado** | el modelo bueno queda sobrescrito | buscar `fit` duplicados después del bucle |
| **Orden de argumentos invertido** | el R² cambia y es incorrecto | siempre `metrica(y_real, y_pred)` |
| **Reentrenar en el apartado de prueba** | la evaluación deja de ser honesta | en 1.A.4 y 1.B.4 solo `predict` |

### Errores que sí lanzan excepción

| Mensaje | Causa | Arreglo |
|---|---|---|
| `NameError: name 'X' is not defined` | notebook ejecutado en desorden | Restart Kernel and Run All Cells |
| `ValueError: too many values to unpack` | falta `.items()` | agregarlo |
| `TypeError: Passing a set as an indexer` | lista escrita con llaves `{ }` | cambiar a corchetes `[ ]` |
| `ValueError: Expected 2D array, got 1D array` | se pasó una Serie como X | doble corchete `df[['col']]`, o la lista de columnas |
| `ValueError: Input contains NaN` | quedaron nulos | `df.isnull().sum()` para localizarlos |
| `ConvergenceWarning` | la logística no convergió | subir `max_iter` |
| `SettingWithCopyWarning` | se modificó una vista, no una copia | `.copy()` al filtrar |

---

## 12. Apartado 1.C — Análisis final

**Qué exige el enunciado:** «analizar cómo influye la cantidad de variables
predictoras en el desempeño y la capacidad de generalización de los modelos,
discutiendo posibles situaciones de *overfitting* o *underfitting*».

**Por qué este apartado decide la nota:** todo lo anterior es ejecución; cualquiera
que siga los pasos obtiene una tabla de métricas. Este apartado es interpretación,
y es donde se distingue a quien entendió el método de quien ejecutó la librería. En
una interrogación oral, las preguntas salen de aquí.

### 12.1 Estructura — cinco apartados

1. **Diagnóstico explícito.** Nombrar *overfitting* o *underfitting* con esas
   palabras, respaldado por las brechas y por los valores absolutos. Decir «el
   desempeño es mejorable» no es un diagnóstico.

2. **Por qué ocurre.** Si es underfitting: ¿el modelo es demasiado simple, o las
   variables no contienen la señal? La correlación máxima de la matriz responde eso.
   Mencionar **qué factores relevantes quedan fuera del dataset**: eso muestra
   comprensión del dominio, no solo del código.

3. **Efecto del número de variables.** Las caídas 10→5 y 5→3, comparadas entre sí.
   Si una cuesta mucho más que la otra, explicar qué dice eso del criterio de
   selección: un ranking univariado mide cada variable por separado y no captura que
   dos variables puedan aportar en conjunto más de lo que aportan sumadas.

4. **Comparación entre las dos tareas.** Regresión y clasificación sobre las mismas
   predictoras, con dificultades distintas. ¿Cuál resultó más difícil y por qué? El
   desbalance de Y₂ entra aquí.

5. **Limitaciones.** Si se usó Pearson, reconocer que solo mide relaciones lineales
   y que la información mutua habría detectado relaciones curvas. Mencionar la
   multicolinealidad si existe entre predictoras.

### 12.2 Evidencia numérica que hay que tener a mano

- [ ] La correlación máxima en valor absoluto entre predictoras y Y₁ (acota el R²)
- [ ] σ de Y₁ en validación, como referencia del RMSE trivial
- [ ] Las tres brechas R²_train − R²_val y su evolución al quitar variables
- [ ] Las caídas de R²_val entre versiones (10→5 y 5→3)
- [ ] La mejora porcentual del modelo sobre predecir siempre la media
- [ ] El cociente RMSE/MAE (≈1,25 si los errores son aproximadamente normales)
- [ ] Las correlaciones máximas **entre predictoras**, para la multicolinealidad
- [ ] La proporción de clases de Y₂ y el accuracy del modelo trivial

### 12.3 La diferencia entre un 5 y un 7

**Nivel 5:** «El R² es bajo, lo que indica que el modelo no explica bien la variable
objetivo.» — Describe el número, no lo explica.

**Nivel 7:** «El R² en validación es bajo y la brecha con entrenamiento es mínima,
lo que descarta sobreajuste e indica underfitting. La causa no es la complejidad del
modelo sino los datos: la correlación más alta entre predictoras y objetivo es *r*,
lo que acota el R² alcanzable por una relación lineal a aproximadamente *r²*. Las
variables disponibles describen propiedades intrínsecas de cada observación,
mientras que el objetivo depende además de factores no presentes en el dataset,
como *…*. Un modelo más flexible sobre las mismas variables no resolvería esto.»

La diferencia es el **mecanismo**: por qué los números son los que son.

### 12.4 Opcional — solo si sobra tiempo

Las curvas de aprendizaje (`learning_curve`) refuerzan el diagnóstico visualmente,
pero el enunciado **no las exige**. Lo mismo para Ridge/Lasso y para
`class_weight='balanced'`. Son lo primero que se recorta.

---

## 13. Checklist antes de entregar tu parte

### Técnico

- [ ] **Restart Kernel and Run All Cells** sin un solo error en rojo
- [ ] Ninguna celda vacía ni con código de prueba olvidado
- [ ] Ninguna celda que imprima miles de filas
- [ ] `seed = 42` y todos los `random_state` intactos
- [ ] Ejecutar dos veces da los mismos números
- [ ] Los gráficos se muestran al ejecutar (`plt.show()`)

### Contenido

- [ ] 1.A.3 tiene los tres párrafos, no solo la tabla
- [ ] 1.A.4 dice **qué métrica** decidió las dos mejores
- [ ] 1.B.1 tiene un ranking distinto al del 1.A.1, con las dos justificaciones
- [ ] 1.B.1 verifica el anidamiento de las sublistas
- [ ] 1.B.3 reporta la proporción de clases y explica por qué el accuracy no sirve
- [ ] 1.B.4 dice qué métrica decidió, coherente con lo argumentado en 1.B.3
- [ ] 1.C nombra explícitamente overfitting o underfitting y da el mecanismo
- [ ] Cada decisión está justificada, no solo descrita

### Coordinación

- [ ] Trabajaste en `tarea2_p1.ipynb`, no en `tarea2.ipynb`
- [ ] Commit y push solo de tu archivo
- [ ] Avisaste cuando terminaste, para unificar

---

## 14. Preguntas de la interrogación oral

Si no puedes responder alguna, ese apartado no está cerrado. Repásalas en voz alta
antes de entregar.

1. ¿Por qué `LinearRegression` no usa gradiente descendiente y
   `LogisticRegression` sí?
2. ¿Por qué escalar no cambia el R² de una regresión lineal pero sí cambia los
   resultados de una logística?
3. Un modelo tiene R²_train = 0,07 y R²_val = 0,068. ¿Overfitting o underfitting?
   ¿Cómo se sabe, y qué habría que mirar para explicar la causa?
4. ¿Por qué el conjunto de prueba se usa una sola vez? ¿Qué se pierde exactamente
   si se usa dos veces?
5. Un modelo tiene accuracy 0,91 y recall 0,02. ¿Qué está pasando, y qué métrica
   habría que reportar?
6. ¿Por qué las sublistas de 10, 5 y 3 variables tienen que ser anidadas?
7. Si la correlación más alta entre predictoras y objetivo es 0,30, ¿qué R² se
   puede esperar como máximo, y por qué?
8. ¿Por qué el ranking de variables para Y₂ es distinto del de Y₁?
9. ¿Qué diferencia hay entre precisión y recall, y cuál mira la fila y cuál la
   columna de la matriz de confusión?
10. ¿Por qué se divide por la desviación estándar al rankear variables para Y₂?
