# Guía de Uso del Proyecto de Clasificación de Texto
Este proyecto consiste en una solución de clasificación de texto que utiliza modelos de lenguaje preentrenados de Hugging Face para clasificar textos en categorías específicas. A continuación, se detalla cómo utilizar y ejecutar este proyecto.

## Requisitos
Antes de comenzar, asegúrese de tener instalados los siguientes requisitos:

- Python (versión recomendada: 3.7+)
- pip (instalador de paquetes de Python)
- Git (opcional, pero útil para clonar el repositorio)
- Acceso a una GPU (opcional, pero recomendado para entrenamiento más rápido)
  
## Instalación

1. Clona este repositorio (o descárgalo como un archivo ZIP y descomprímelo):
```
git clone https://github.com/rfeuereisen/DistilBERT-Text_Classifier.git
cd DistilBERT-Text_Classifier
```

2. Crea un entorno virtual para el proyecto (opcional pero recomendado):
```
python -m venv venv
source venv/bin/activate  # Para sistemas Unix / Linux
venv\Scripts\activate  # Para Windows
```

3. Instale las dependencias del proyecto:
```
pip install -r requirements.txt
```

## Configuración
El proyecto contiene varios archivos de configuración que necesitas ajustar antes de su uso. Asegúrate de configurar lo siguiente:

- `generate_datasets.py`: Este archivo se utiliza para generar los conjuntos de entrenamiento y prueba. Asegúrate de ajustar `CATEGORY` al tipo de clasificación que deseas (p. ej., "nivel_3"). Genera los archivos `{categoria}_categories.json`, `train_{categoria}_filtered.csv` y `test_{categoria}_filtered.csv`
- `train.py`: Defina la configuración del entrenamiento, como el modelo preentrenado a finetunear y los hiperparámetros de entrenamiento. Asegúrese de ajustar `MODEL_TO_TRAIN`, `MODEL_CKPT`, `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, y otros valores según sus necesidades. Es recomentable para nivel 1 y tipo de detención entrenar con 3 épocas y para nivel 2, nivel 3 y comentario entrenar con 5 épocas

## Generación de Datos
Antes de entrenar el modelo, necesitas generar los conjuntos de entrenamiento y prueba. Ejecuta el siguiente comando:
```
python generate_datasets.py
```
Esto generará los archivos CSV de entrenamiento y prueba en la carpeta data.

## Entrenamiento del modelo
Para entrenar el modelo, ejecuta el siguiente comando:
```
python train.py
```
El entrenamiento puede llevar un tiempo, especialmente si utiliza una GPU. Luego del entrenamiento, deberían guardarse 8 archivos por modelo.

## Inferencia
Una vez entrenado el modelo, se hace la clasificación ejecutando el archivo `inference.py`, este abre una interfaz gráfica que pregunta cual es el archivo Excel a clasificar y luego de la clasificación esta pregunta donde se quiere guardar el archivo de resultados. Además, puedes realizar inferencias en textos específicos. Puedes hacerlo ejecutando el código proporcionado en `test.ipynb`. Este cuaderno de Jupyter te permitirá cargar el modelo, realizar predicciones y visualizar la matriz de confusión.
