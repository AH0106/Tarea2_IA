Realización de la tarea 2 de IA.

Se ocuparan 2 datasets diferentes para la pregunta 1 y la pregunta 2, respectivamente


Para la pregunta 1, el dataset a utilizar sera:
https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m


Para la pregunta 2, el dataset a utilizar sera:
https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17


Librerias a importar:

numpy 
pandas 
matplotlib.pyplot 
seaborn


Este proyecto contiene el análisis exploratorio, preprocesamiento y modelado de datos para dos conjuntos de datos: SDSS17 (Clasificación Estelar de Objetos Celestiales) y Goodreads (Análisis de Libros y Reseñas).

Requisitos Previos

Asegúrate de tener instalado lo siguiente en tu sistema antes de comenzar:

Python: Versión 3.9 o superior.

Git: Para clonar el repositorio.

Pip: Gestor de paquetes de Python.

Guía de Instalación y Configuración

Sigue estos pasos detallados para dejar listo el entorno de desarrollo local.

1. Clonar el Repositorio

Abre tu terminal o línea de comandos y ejecuta:

git clone <URL_DE_TU_REPOSITORIO>
cd <NOMBRE_DE_LA_CARPETA_DEL_PROYECTO>


2. Crear y Activar un Entorno Virtual

Es recomendable crear un entorno virtual para mantener aisladas las dependencias del proyecto.

python3 -m venv venv
source venv/bin/activate


3. Instalar las Dependencias

Ejecuta el siguiente comando para instalar automáticamente todas las bibliotecas requeridas:

pip install --upgrade pip
pip install -r requirements.txt


pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab


Ubicación de los Datos (Datasets)



├── data/
│   ├── sdss17.csv         # Conjunto de datos de Clasificación Estelar SDSS17
│   └── goodreads.csv      # Conjunto de datos de Libros de Goodreads
├── notebooks/
│   └── analisis_sdss17_goodreads.ipynb
├── requirements.txt
└── README.md




Tecnologías y Librerías Principales

Pandas & NumPy: Manipulación y análisis sintáctico de estructuras de datos.

Matplotlib & Seaborn: Visualización estática de datos y gráficos exploratorios.

Scikit-Learn: Algoritmos de machine learning, preprocesamiento y métricas de evaluación.

JupyterLab / Notebook: Entorno interactivo de desarrollo.

