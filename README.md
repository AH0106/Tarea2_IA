# Tarea 2 de Inteligencia Artificial

- **Pregunta 1 · Spotify.** Se usan características de las canciones para predecir su popularidad con regresión lineal y clasificar su contenido explícito con regresión logística. Se compara el desempeño con 10, 5 y 3 predictores. [Notebook de la Pregunta 1](notebooks/tarea2_p1.ipynb)
- **Pregunta 2 · SDSS17 (en desarrollo).** Se busca comparar la clasificación de galaxias, estrellas y cuásares con el 10 % o el 30 % de las etiquetas de entrenamiento conocidas, y medir qué cambia al incorporar pseudoetiquetas. [Notebook, Parte 2](notebooks/tarea2.ipynb) · [Plan de trabajo](README_Pregunta2.md)

## Tecnologías

Python y JupyterLab para ejecutar los notebooks, pandas y NumPy para preparar los datos, Matplotlib y seaborn para visualizarlos y scikit-learn para entrenar y evaluar los modelos.

## Ejecución local

Con Git y Python 3.12, ejecuta estos comandos en macOS o Linux.

```bash
git clone --branch main3 https://github.com/AH0106/Tarea2_IA.git
cd Tarea2_IA
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd notebooks
python -m jupyterlab
```

Los dos CSV están incluidos en [notebooks/data/raw](notebooks/data/raw). En JupyterLab, elige el notebook según la pregunta.

1. **Spotify.** Abre `tarea2_p1.ipynb` y ejecuta todas las celdas en orden.
2. **SDSS17.** Abre `tarea2.ipynb`. En «0. Configuración inicial», ejecuta la celda de importaciones y la de carga local de los CSV. Omite la descarga de Kaggle y salta a «Parte 2» para ejecutar las celdas implementadas en orden.
