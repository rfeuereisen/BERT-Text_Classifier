from pathlib import Path
import sys
import os
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# Specify the path of the model
MODEL_CKPT_NIVEL_1 = Path(
    "./models/dccuchile/distilbert-base-spanish-uncased-finetuned-nivel_1/"
)

MODEL_CKPT_NIVEL_2 = Path(
    "./models/dccuchile/distilbert-base-spanish-uncased-finetuned-nivel_2/"
)

MODEL_CKPT_NIVEL_3 = Path(
    "./models/dccuchile/distilbert-base-spanish-uncased-finetuned-nivel_3/"
)

MODEL_CKPT_COMENTARIO = Path(
    "./models/dccuchile/distilbert-base-spanish-uncased-finetuned-comentario/"
)

MODEL_CKPT_NIVEL_TIPO_DETENCION = Path(
    "./models/dccuchile/distilbert-base-spanish-uncased-finetuned-tipo_detencion/"
)
# Load the fine-tuned tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT_NIVEL_1)
model_nivel_1 = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT_NIVEL_1).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT_NIVEL_2)
model_nivel_2 = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT_NIVEL_2).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT_NIVEL_3)
model_nivel_3 = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT_NIVEL_3).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT_COMENTARIO)
model_comentario = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT_COMENTARIO).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT_NIVEL_TIPO_DETENCION)
model_tipo_detencion = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT_NIVEL_TIPO_DETENCION).to(device)




def inference(text: str, model) -> str:
    "Predict the category of a given text"
    inputs = tokenizer(text, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]

    max_vale = max(predictions)
    idx = predictions.index(max_vale)
    return model.config.id2label[idx]


if __name__ == "__main__":
    df = pd.read_excel("data/prueba2.xlsx", sheet_name='BD 2')
    colum_texts = list(df['Datos BI'])


    for idx, text in enumerate(colum_texts):

        numero_reason = df.loc[idx, 'Número Reason']
        print(f"Procesando índice {idx}: Número Reason = {numero_reason}")
        # Condición 1: '3000'
        if str(df.loc[idx, 'Número Reason']) == '3000':
            print("Condición 1 se cumple")
            df.loc[idx, 'Nivel_1'] = 'PM'
            df.loc[idx, 'Nivel_2'] = 'PM'
            df.loc[idx, 'Nivel_3'] = None
            df.loc[idx, 'Comentario'] = 'PM'
            df.loc[idx, 'Tipo de Detención'] = 'MP'

        # Condición 2: '3030'
        if str(df.loc[idx, 'Número Reason']) == '3030':
            print("Condición 2 se cumple")
            df.loc[idx, 'Nivel_1'] = 'NEUMATICOS_Y_AROS'
            df.loc[idx, 'Nivel_2'] = 'NEUMATICOS_Y_AROS'
            df.loc[idx, 'Nivel_3'] = None
            df.loc[idx, 'Comentario'] = 'PM'
            df.loc[idx, 'Tipo de Detención'] = 'MPV'

        # Condición 3: '5030'
        if str(df.loc[idx, 'Número Reason']) == '5030':
            print("Condición 3 se cumple")
            df.loc[idx, 'Nivel_1'] = 'NEUMATICOS_Y_AROS'
            df.loc[idx, 'Nivel_2'] = 'NEUMATICOS_Y_AROS'
            df.loc[idx, 'Nivel_3'] = None
            category_comentario = inference(str(text), model_comentario)
            df.loc[idx, 'Comentario'] = category_comentario
            df.loc[idx, 'Tipo de Detención'] = 'MCV'

        # Condición 4: '5010'
        if str(df.loc[idx, 'Número Reason']) == '5010':
            print("Condición 4 se cumple")
            category_nivel_1 = inference(str(text), model_nivel_1)
            if category_nivel_1 == "VACIO":
                category_nivel_1 = ""
            df.loc[idx, 'Nivel_1'] = category_nivel_1
            category_nivel_2 = inference(str(text), model_nivel_2)
            if category_nivel_2 == "VACIO":
                category_nivel_2 = ""
            df.loc[idx, 'Nivel_2'] = category_nivel_2
            category_nivel_3 = inference(str(text), model_nivel_3)
            if category_nivel_3 == "VACIO":
                category_nivel_3 = ""
            df.loc[idx, 'Nivel_3'] = category_nivel_3
            category_comentario = inference(str(text), model_comentario)
            df.loc[idx, 'Comentario'] = category_comentario
            df.loc[idx, 'Tipo de Detención'] = 'MCA'

        # Condición 5: '5020'
        if str(df.loc[idx, 'Número Reason']) == '5020':
            print("Condición 5 se cumple")
            category_nivel_1 = inference(str(text), model_nivel_1)
            if category_nivel_1 == "VACIO":
                category_nivel_1 = ""
            df.loc[idx, 'Nivel_1'] = category_nivel_1
            category_nivel_2 = inference(str(text), model_nivel_2)
            if category_nivel_2 == "VACIO":
                category_nivel_2 = ""
            df.loc[idx, 'Nivel_2'] = category_nivel_2
            category_nivel_3 = inference(str(text), model_nivel_3)
            if category_nivel_3 == "VACIO":
                category_nivel_3 = ""
            df.loc[idx, 'Nivel_3'] = category_nivel_3
            category_comentario = inference(str(text), model_comentario)
            df.loc[idx, 'Comentario'] = category_comentario
            df.loc[idx, 'Tipo de Detención'] = 'MCI'

        # Condición 6: '5050'
        if str(df.loc[idx, 'Número Reason']) == '5050':
            print("Condición 6 se cumple")
            category_nivel_1 = inference(str(text), model_nivel_1)
            if category_nivel_1 == "VACIO":
                category_nivel_1 = ""
            df.loc[idx, 'Nivel_1'] = category_nivel_1
            category_nivel_2 = inference(str(text), model_nivel_2)
            if category_nivel_2 == "VACIO":
                category_nivel_2 = ""
            df.loc[idx, 'Nivel_2'] = category_nivel_2
            category_nivel_3 = inference(str(text), model_nivel_3)
            if category_nivel_3 == "VACIO":
                category_nivel_3 = ""
            df.loc[idx, 'Nivel_3'] = category_nivel_3
            category_comentario = inference(str(text), model_comentario)
            df.loc[idx, 'Comentario'] = category_comentario
            df.loc[idx, 'Tipo de Detención'] = 'MCM'

        # Condición 7: '5070'
        if str(df.loc[idx, 'Número Reason']) == '5070':
            print("Condición 7 se cumple")
            category_nivel_1 = inference(str(text), model_nivel_1)
            if category_nivel_1 == "VACIO":
                category_nivel_1 = ""
            df.loc[idx, 'Nivel_1'] = category_nivel_1
            category_nivel_2 = inference(str(text), model_nivel_2)
            if category_nivel_2 == "VACIO":
                category_nivel_2 = ""
            df.loc[idx, 'Nivel_2'] = category_nivel_2
            category_nivel_3 = inference(str(text), model_nivel_3)
            if category_nivel_3 == "VACIO":
                category_nivel_3 = ""
            df.loc[idx, 'Nivel_3'] = category_nivel_3
            category_comentario = inference(str(text), model_comentario)
            df.loc[idx, 'Comentario'] = category_comentario
            df.loc[idx, 'Tipo de Detención'] = 'MCI'

        # Condición 8: '5000'
        if str(df.loc[idx, 'Número Reason']) == '5000':
            print("Condición 8 se cumple")
            category_nivel_1 = inference(str(text), model_nivel_1)
            if category_nivel_1 == "VACIO":
                category_nivel_1 = ""
            df.loc[idx, 'Nivel_1'] = category_nivel_1
            category_nivel_2 = inference(str(text), model_nivel_2)
            if category_nivel_2 == "VACIO":
                category_nivel_2 = ""
            df.loc[idx, 'Nivel_2'] = category_nivel_2
            category_nivel_3 = inference(str(text), model_nivel_3)
            if category_nivel_3 == "VACIO":
                category_nivel_3 = ""
            df.loc[idx, 'Nivel_3'] = category_nivel_3
            category_comentario = inference(str(text), model_comentario)
            df.loc[idx, 'Comentario'] = category_comentario
            category_tipo = inference(str(text), model_tipo_detencion)
            df.loc[idx, 'Tipo de Detención'] = category_tipo
            

    # Verificar si la carpeta 'data' existe, y si no, crearla
    if not os.path.exists('data'):
        os.makedirs('data')

    # Guardar el DataFrame en un archivo de Excel en la carpeta 'data'
    df.to_excel('data/Resultados.xlsx', index=False)
