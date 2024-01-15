# Guía de Uso del Proyecto de Clasificación de Texto

Este proyecto ofrece una solución de clasificación de texto que emplea modelos de lenguaje preentrenados de Hugging Face para categorizar textos en grupos específicos. A continuación, se detalla cómo utilizar y ejecutar este proyecto, así como la función de cada archivo en la estructura del proyecto.

## Requisitos

Antes de comenzar, asegúrese de tener instalados los siguientes requisitos:

- Python (versión recomendada: 3.7+)
- pip (instalador de paquetes de Python)
- Git (opcional, pero útil para clonar el repositorio)
- Acceso a una GPU (opcional, pero recomendado para entrenamiento más rápido)
  
## Instalación

Instala las dependencias del proyecto ejecutando el siguiente comando:
```
pip install -r requirements.txt
```

## Configuración

El proyecto incluye varios archivos de configuración que debes ajustar antes de utilizarlo. Asegúrate de configurar lo siguiente:

- `generate_datasets.py`: Este archivo se utiliza para generar conjuntos de entrenamiento y prueba. Ajusta la variable CATEGORY al tipo de clasificación que deseas (por ejemplo, `nivel_3`). Genera archivos como `{categoria}_categories.json`, `train_{categoria}_filtered.csv` y `test_{categoria}_filtered.csv`.
- `train.py`: Define la configuración del entrenamiento, como el modelo preentrenado a finetunear y los hiperparámetros de entrenamiento. Asegúrese de ajustar `MODEL_TO_TRAIN`, `MODEL_CKPT`, `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, y otros valores según sus necesidades. Es recomentable para nivel 1 y tipo de detención entrenar con 3 épocas y para nivel 2, nivel 3 y comentario entrenar con 5 épocas

## Generación de Datos y Entrenamiento del modelo

Antes de entrenar el modelo, es necesario generar conjuntos de entrenamiento y prueba para cada categoría específica. Esto se logra ejecutando `generate_datasets.py` múltiples veces, ajustando la variable `CATEGORY` al tipo de clasificación deseado. Por ejemplo, para entrenar modelos en las categorías nivel_1, nivel_2, etc., se deben ejecutar los siguientes comandos:

```
python generate_datasets.py --category nivel_1
python generate_datasets.py --category nivel_2
# ... y así sucesivamente para cada categoría deseada
```

Este proceso generará archivos CSV de entrenamiento y prueba para cada categoría en la carpeta `data`. Después de generar los conjuntos de datos, es hora de entrenar los modelos. Para esto, ejecuta `train.py` ajustando la variable MODEL_TO_TRAIN en consecuencia.

```
python train.py --category nivel_1
python train.py --category nivel_2
# ... y así sucesivamente para cada categoría deseada
```
El entrenamiento puede llevar tiempo, especialmente con GPU. Después del entrenamiento, se guardarán 8 archivos por modelo en la carpeta `models`.

## Inferencia

Una vez entrenado el modelo, realiza clasificaciones ejecutando `inference.py` para clasificar de manera local y `inference2.py` si se quiere utilizar docker para clasificar de forma web. Este script abre una interfaz gráfica para clasificar un archivo Excel y guardar los resultados. También puedes realizar inferencias en textos específicos ejecutando el código proporcionado en `test.ipynb`.

## Estrucutra de archivos

- data: Contiene archivos de datos generados por `generate_datasets.py`.
- entry_point: Contiene scripts y utilidades para funciones específicas.
- models: Almacena modelos entrenados.
- src: Contiene archivos de código fuente esenciales.
- train_model: Incluye archivos relacionados con el proceso de entrenamiento.

## Descripción de archivos

### `entry_point/feedback.py`

Código para corregir manualmente clasificaciones incorrectas en un conjunto de datos.

### `entry_point/generate_datasets.py`

Este script genera conjuntos de entrenamiento y prueba a partir del conjunto de datos crudo (por ejemplo, `nivel_3.csv`). Limpia y preprocesa los datos, crea un archivo de categorías (`nivel_3_categories.json`) y guarda los conjuntos de entrenamiento y prueba filtrados (`train_nivel_3_filtered.csv` y `test_nivel_3_filtered.csv`, respectivamente).

### `entry_point/inference.py`

Este script realiza inferencia en textos utilizando modelos preentrenados para diferentes niveles y tipos de detención. Los modelos utilizados están ubicados en las rutas especificadas en `MODEL_CKPT_NIVEL_1`, `MODEL_CKPT_NIVEL_2`, etc. La inferencia se realiza en un archivo de Excel seleccionado por el usuario.

### `entry_point/train.py`

Entrena un modelo de clasificación para una categoría específica (`nivel_1`, `nivel_2`, etc.). Utiliza un modelo preentrenado (`distilbert-base-spanish-uncased`) y el conjunto de datos de entrenamiento creado por `generate_datasets.py`. Guarda el modelo entrenado en la carpeta models con un nombre específico (`dccuchile/distilbert-base-spanish-uncased-finetuned-nivel_1`, etc.).

### `models/`

Contiene modelos entrenados para cada tipo de clasificación.

### `src/config.py`

Define la configuración del modelo, incluidos los mapeos de etiquetas y la ubicación del modelo preentrenado. Utilizado por `model.py` y `train.py`.

### `src/load_dataset.py`

Carga conjuntos de datos desde archivos CSV y crea un conjunto de datos para el entrenamiento y prueba. Utiliza la biblioteca Hugging Face Datasets. Utilizado por `train.py`.

### `src/model.py`

Define la clase del modelo de clasificación (Classifier). Inicializa el modelo, exporta métricas y calcula métricas de evaluación. Utilizado por `train.py`.

### `src/tokenize_dataset.py`

Proporciona una clase (TokenizeDataset) para tokenizar conjuntos de datos utilizando un modelo preentrenado. Utilizado por `classifier_trainer.py`.

### `train_model/classifier_trainer.py`

Contiene la clase SequenceClassifier, que orquesta la carga de datos, tokenización, entrenamiento del modelo y exportación de métricas. Utiliza clases y funciones de `model.py`, `load_dataset.py`, y `tokenize_dataset.py`.

### Notebooks ( `data_exploration.ipynb` y `test.ipynb`)

Proporcionan análisis exploratorio de datos y pruebas de inferencia en un modelo previamente entrenado, respectivamente. Utilizan archivos CSV y modelos guardados.

En resumen, los scripts y cuadernos están diseñados para trabajar en conjunto, donde `generate_datasets.py` prepara los datos, `train.py` entrena modelos, `inference.py` realiza inferencias, y los cuadernos (`data_exploration.ipynb` y `test.ipynb`) proporcionan análisis y pruebas adicionales. La modularidad y la interconexión entre estos archivos permiten un flujo de trabajo estructurado para el procesamiento de datos y la construcción de modelos de clasificación.
