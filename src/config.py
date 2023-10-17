from typing import Dict

from transformers import AutoConfig


class Appropriateness:
    """Class representing the configuration"""

    def __init__(
        self,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        num_labels: int,
        model_ckpt: str,
    ):
        # Inicializa los atributos de la clase con los valores proporcionados
        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = num_labels
        self.model_ckpt = model_ckpt
        self.custom_config = self.get_config()

    def get_config(self):
        """Method to get the configuration"""

        # Utiliza AutoConfig para crear una configuración personalizada
        return AutoConfig.from_pretrained(
            self.model_ckpt,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )


'''
Este archivo define una clase que facilita la configuración de un modelo de clasificación de texto. 
La clase toma como entrada diccionarios que mapean etiquetas a identificadores y viceversa, 
el número de etiquetas en el problema de clasificación y la ruta del punto de control (checkpoint) del modelo. 
Luego, utiliza la clase AutoConfigde Transformers para generar una configuración personalizada para el modelo, 
que incluye la información de etiquetas e identificadores. 
Esta configuración personalizada se almacena en el atributo custom_configde la instancia de la clase Appropriateness.
'''