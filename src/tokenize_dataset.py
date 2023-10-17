from datasets.dataset_dict import DatasetDict
from datasets.formatting.formatting import LazyBatch
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class TokenizeDataset():
    """Class representing a tokenizer of datasets"""
    def __init__(self, model_ckpt: str):

        # Inicializa la clase con la ruta del punto de control del modelo
        self.model_ckpt = model_ckpt

        # Carga el tokenizador adecuado para el modelo preentrenado
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)

    def encoded_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Method to encode the dataset"""

        # Aplica la tokenización al conjunto de datos, utilizando el método _tokenize
        return dataset.map(self._tokenize, batched=True, batch_size=None)

    def _tokenize(self, batch: LazyBatch) -> BatchEncoding:
        """Method to tokenize the texts by batches"""

        # Utiliza el tokenizador para procesar los textos en el lote
        return self.tokenizer(batch['text'], padding=True, truncation=True)


'''
Este archivo define una clase que utiliza un tokenizador preentrenado (por ejemplo, DistilBERT) para procesar los textos en un conjunto de datos. 
El método encoded_datasettoma un conjunto de datos de entrada y aplica la tokenización a través del método _tokenize, 
que utiliza el tokenizador y realiza el procesamiento en lotes. 
El resultado es un conjunto de datos tokenizado que se puede utilizar para entrenar un modelo de clasificación de texto.
'''