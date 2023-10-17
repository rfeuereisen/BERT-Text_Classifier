from pathlib import Path
from typing import Tuple, Dict
import os

import pandas as pd
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from datasets.formatting.formatting import LazyRow


class LoadDataset:
    """Class representing a Dataset"""
    def __init__(self, path_dataset_train: Path, path_dataset_test: Path, dict_int2str: dict):

        # Inicializa los atributos de la clase con las rutas de los conjuntos de datos y un diccionario de mapeo
        self.path_dataset_train = path_dataset_train
        self.path_dataset_test = path_dataset_test
        self.dict_int2str = dict_int2str
        self.dict_str2int = {value: key for (key, value) in self.dict_int2str.items()}
        self.num_labels = len(self.dict_int2str)

    def load_dataset(self) -> Tuple[pd.DataFrame]:
        """Method to load dataset in Data Frame format"""

        # Lee los conjuntos de datos de entrenamiento y prueba desde archivos CSV
        train = pd.read_csv(self.path_dataset_train, encoding="ISO-8859-1")
        test = pd.read_csv(self.path_dataset_test, encoding="ISO-8859-1")
        return train, test

    def create_dataset(self, train: pd.DataFrame, test: pd.DataFrame) -> DatasetDict:
        """Method to create useful datasets for training using Hugging Face"""

        # Crea diccionarios con los conjuntos de datos
        dict_data = {"train": train, "valid": test}

        # Guarda temporariamente los DataFrames en archivos CSV
        for key, value in dict_data.items():
            df_temp = value
            df_temp.to_csv(f"temp_{key}.csv", index=False)

        # Carga los datos utilizando la biblioteca 'datasets' de Hugging Face
        data_files = {"train": "temp_train.csv", "test": "temp_valid.csv"}
        dataset = load_dataset(
            "csv", data_files={"train": "temp_train.csv", "test": "temp_valid.csv"}
        )

        # Limpia los archivos temporales
        self._clean_temp_files(data_files)

        # Mapea las etiquetas a sus representaciones de cadena
        dataset = dataset.map(self._create_label_str)
        return dataset

    def _create_label_str(self, batch: LazyRow) -> Dict[str, str]:
        return {"label_name": self.dict_int2str[batch["label"]]}

    def _clean_temp_files(self, data_files: Dict[str, str]) -> None:
        files = list(data_files.values())
        for file in files:
            os.remove(file)


'''
Este script carga y formatea los datos de entrenamiento y prueba, los almacena temporalmente en archivos CSV, 
utiliza la biblioteca 'datasets' de Hugging Face para cargar los datos, 
mapea las etiquetas a sus representaciones de cadena y finalmente limpia los archivos temporales creados. 
Este proceso es útil para preparar los datos de entrenamiento en un formato compatible con la biblioteca de Transformers de Hugging Face.
'''
