from pathlib import Path

from transformers import Trainer

from src.config import Appropriateness
from src.load_dataset import LoadDataset
from src.model import Classifier
from src.tokenize_dataset import TokenizeDataset

class SequenceClassifier():
    """Class representing a multi-class classifier"""
    def __init__(
        self,
        model_ckpt: str,
        train_dataset: str,
        test_dataset: str,
        dict_int2str: dict,
    ) -> None:

        # Inicializa la clase con la ruta del modelo preentrenado, los conjuntos de datos y el diccionario de mapeo
        self.model_ckpt = model_ckpt
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dict_int2str = dict_int2str
        self.dict_str2int = {value: key for (key, value) in self.dict_int2str.items()}
        self.num_labels = len(self.dict_int2str)
        self.path_dataset_train = (
            Path("data") / self.train_dataset
        )
        self.path_dataset_test = (
            Path("data") / self.test_dataset
        )

    def get_dataset(self):
        """Method to get the dataset needed for the training"""

        # Carga y tokeniza los conjuntos de datos
        loader = LoadDataset(self.path_dataset_train, self.path_dataset_test, self.dict_int2str)
        train, test = loader.load_dataset()
        train.dropna(inplace=True)
        test.dropna(inplace=True)
        dataset = loader.create_dataset(train, test)
        tokenized_dataset = self._tokenize_dataset(dataset)
        return tokenized_dataset

    def train_classifier(self, training_args, dataset_tokenized, model_name) -> None:
        """Method to train the choosen classifier"""

        # Configura el tokenizador, el modelo y el entrenador
        tokenizer = TokenizeDataset(self.model_ckpt)
        model = self._get_model()
        trainer = Trainer(
            model=model.model,
            args=training_args,
            compute_metrics=model.compute_metrics,
            train_dataset=dataset_tokenized["train"],
            eval_dataset=dataset_tokenized["test"],
            tokenizer=tokenizer.tokenizer,
        )

        # Inicia el entrenamiento
        trainer.train()
        trainer.save_model(model_name)

        # Exporta las métricas del conjunto de prueba
        model.export_metrics(
            trainer,
            dataset_tokenized["test"],
            f"{model_name}/multiclass_test_metrics.csv",
        )

    def _tokenize_dataset(self, dataset):

        # Tokeniza el conjunto de datos
        tokenizer = TokenizeDataset(self.model_ckpt)
        dataset_tokenized = tokenizer.encoded_dataset(dataset)
        dataset_tokenized.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )
        return dataset_tokenized

    def _get_config(self):
        return Appropriateness(
            id2label=self.dict_int2str,
            label2id=self.dict_str2int,
            num_labels=self.num_labels,
            model_ckpt=self.model_ckpt,
        )

    def _get_model(self):

        # Obtiene la configuración y el modelo
        config = self._get_config()
        return Classifier(
            config=config.custom_config, model_ckpt=self.model_ckpt
        )


'''
Este archivo define una clase que se encarga de coordinar el proceso de entrenamiento de un modelo de clasificación de texto. 
La clase carga los datos, tokeniza los textos, configura el modelo y el tokenizador, 
y finalmente inicia el entrenamiento y guarda el modelo entrenado. También exporta métricas en un archivo CSV.
'''