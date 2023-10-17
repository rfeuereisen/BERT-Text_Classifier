from pathlib import Path
import sys
import json

from transformers import TrainingArguments

# Establecer el directorio raíz del proyecto
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from train_model.classifier_trainer import (
    SequenceClassifier
)

# Definir la categoría de modelo que se entrenará
MODEL_TO_TRAIN = "nivel_1" #tipo_detencion, nivel_1, nivel_2, nivel_3, comentario
MODEL_CKPT = "dccuchile/distilbert-base-spanish-uncased"

# Nombres de los conjuntos de datos de entrenamiento y prueba
TRAIN_DATASET = f"train_{MODEL_TO_TRAIN}_filtered.csv"
TEST_DATASET = f"test_{MODEL_TO_TRAIN}_filtered.csv"

# Cargar el diccionario de categorías desde un archivo JSON
with open(f'data/{MODEL_TO_TRAIN}_categories.json', 'r') as openfile:
    CATEGORIES = json.load(openfile)
    CATEGORIES = {int(k): v for k, v in CATEGORIES.items()}


# Hiperparámetros para el entrenamiento del modelo
BATCH_SIZE = 32
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Función principal
def main() -> None:

    # Crear una instancia del SequenceClassifier con el modelo, los conjuntos de datos y las categorías
    classifier = SequenceClassifier(
                MODEL_CKPT,
                TRAIN_DATASET,
                TEST_DATASET,
                CATEGORIES,
            )
    
    # Tokenizar el conjunto de datos
    dataset_tokenized = classifier.get_dataset()

    # Calcular el número de pasos de registro durante el entrenamiento
    logging_steps = len(dataset_tokenized["train"]) // BATCH_SIZE

    # Definir el nombre del modelo de salida
    model_name = (
        f"models/{MODEL_CKPT}-finetuned-{MODEL_TO_TRAIN}"
    )

    # Configurar los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=model_name,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=False,
        log_level="error",
    )

    # Entrenar el clasificador
    classifier.train_classifier(training_args, dataset_tokenized, model_name)


if __name__ == '__main__':
    main()


'''
Este script carga un modelo preentrenado, lo adapta a la tarea de clasificación de texto, 
y lo entrena utilizando los datos de entrenamiento y los hiperparámetros especificados. 
Los resultados del entrenamiento se guardan en una carpeta llamada "modelos".
'''
