from typing import Dict

import pandas as pd

from datasets.arrow_dataset import Dataset
import torch
from transformers import AutoModelForSequenceClassification
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import accuracy_score, f1_score


class Classifier:
    """Class representing a appropriateness classifier"""

    def __init__(self, config: DistilBertConfig, model_ckpt: str = ""):

        # Inicializa la clase con la configuración y la ruta del punto de control del modelo
        self.config = config
        self.model_ckpt = model_ckpt

        # Determina si se usará la GPU (si está disponible)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Carga el modelo preentrenado desde el punto de control
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_ckpt, config=self.config
        ).to(self.device)

    def export_metrics(
        self, trainer: Trainer, dataset_test: Dataset, output_path: str
    ) -> None:
        """Method to export the metrics at the end of the training"""

        # Realiza predicciones en el conjunto de prueba utilizando el entrenador
        preds_output = trainer.predict(dataset_test)
        preds_output = {
            metric: [value] for (metric, value) in preds_output.metrics.items()
        }
        df_metrics = pd.DataFrame(preds_output)
        df_metrics.to_csv(output_path, index=False)

    def compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """Method to compute the metrics"""

        # Calcula las métricas de precisión y F1 a partir de las predicciones y etiquetas reales
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}
    

'''
Este archivo define una clase que se encarga de cargar un modelo preentrenado (en este caso, DistilBERT) desde un punto de control, 
gestionar la GPU si está disponible, exportar métricas después del entrenamiento y calcular métricas de precisión y F1. 
El modelo de clasificación se utiliza para realizar predicciones en texto. La clase Classifierse encarga de facilitar la gestión y evaluación del modelo.
'''
