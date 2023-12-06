from pathlib import Path
import sys
import os
import pandas as pd
from tkinter import Tk, filedialog, simpledialog

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


def select_excel_and_sheet():
    root = Tk()
    root.withdraw()

    # Seleccionar el archivo de Excel
    excel_file_path = filedialog.askopenfilename(title="Selecciona el archivo de Excel",
                                                 filetypes=[("Excel files", "*.xlsx;*.xls")])

    if not excel_file_path:
        print("No se seleccionó un archivo. Saliendo.")
        exit()

    # Leer las pestañas disponibles
    df = pd.read_excel(excel_file_path, sheet_name=None)
    sheet_names = list(df.keys())

    if not sheet_names:
        print("El archivo de Excel no contiene pestañas. Saliendo.")
        exit()

    # Seleccionar la pestaña
    sheet_name = simpledialog.askstring("Selecciona la pestaña de Excel", "Pestañas disponibles: " + ", ".join(sheet_names))

    if sheet_name not in sheet_names:
        print("Pestaña no válida. Saliendo.")
        exit()

    return excel_file_path, sheet_name

def save_results():
    root = Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx;*.xls")])
    return file_path

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

    # Seleccionar el archivo de entrada y la pestaña de Excel
    excel_file_path, sheet_name = select_excel_and_sheet()

    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    colum_texts = list(df['Datos BI'])


    for idx, text in enumerate(colum_texts):

        numero_reason = df.loc[idx, 'Número Reason']
        print(f"Procesando índice {idx}: Número Reason = {numero_reason}")

        # Condición 1: '3000'
        if str(df.loc[idx, 'Número Reason']) == '3000':
            print("Condición 1 se cumple")
            category_comentario = inference(str(text), model_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
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
            category_tipo = inference(str(text), model_tipo_detencion)
            if category_tipo == "VACIO":
                category_tipo = ""
            df.loc[idx, 'Tipo de Detención'] = category_tipo
        
         # Condición 2: '3010'
        if str(df.loc[idx, 'Número Reason']) == '3010':
            print("Condición 2 se cumple")
            category_comentario = inference(str(text), model_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
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
            category_tipo = inference(str(text), model_tipo_detencion)
            if category_tipo == "VACIO":
                category_tipo = ""
            df.loc[idx, 'Tipo de Detención'] = category_tipo

        # Condición 3: '3030'
        if str(df.loc[idx, 'Número Reason']) == '3030':
            print("Condición 3 se cumple")
            category_comentario = inference(str(text), model_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
            df.loc[idx, 'Nivel_1'] = 'NEUMATICOS'
            df.loc[idx, 'Nivel_2'] = 'NEUMATICOS_Y_AROS'
            df.loc[idx, 'Nivel_3'] = None
            df.loc[idx, 'Tipo de Detención'] = "MPV"

        # Condición 4: '5000'
        if str(df.loc[idx, 'Número Reason']) == '5000':
            print("Condición 4 se cumple")
            category_comentario = inference(str(text), model_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
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
            category_tipo = inference(str(text), model_tipo_detencion)
            if category_tipo == "VACIO":
                category_tipo = ""
            df.loc[idx, 'Tipo de Detención'] = category_tipo

        # Condición 5: '5010'
        if str(df.loc[idx, 'Número Reason']) == '5010':
            print("Condición 5 se cumple")
            category_comentario = inference(str(text), model_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
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
            df.loc[idx, 'Tipo de Detención'] = 'MCA'

         # Condición 6: '5020'
        if str(df.loc[idx, 'Número Reason']) == '5020':
            print("Condición 6 se cumple")
            category_comentario = inference(str(text), model_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
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
            category_tipo = inference(str(text), model_tipo_detencion)
            if category_tipo == "VACIO":
                category_tipo = ""
            df.loc[idx, 'Tipo de Detención'] = category_tipo

        # Condición 7: '5030'
        if str(df.loc[idx, 'Número Reason']) == '5030':
            print("Condición 7 se cumple")
            category_comentario = inference(str(text), model_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
            df.loc[idx, 'Nivel_1'] = 'NEUMATICOS_Y_AROS'
            df.loc[idx, 'Nivel_2'] = 'NEUMATICOS_Y_AROS'
            df.loc[idx, 'Nivel_3'] = None
            df.loc[idx, 'Tipo de Detención'] = 'MCV'

        # Condición 8: '5050'
        if str(df.loc[idx, 'Número Reason']) == '5050':
            print("Condición 8 se cumple")
            category_comentario = inference(str(text), model_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
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
            df.loc[idx, 'Tipo de Detención'] = 'MCM'

        # Condición 9: '5070'
        if str(df.loc[idx, 'Número Reason']) == '5070':
            print("Condición 9 se cumple")
            category_comentario = inference(str(text), model_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
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
            category_tipo = inference(str(text), model_tipo_detencion)
            if category_tipo == "VACIO":
                category_tipo = ""
            df.loc[idx, 'Tipo de Detención'] = category_tipo
            

    # Verificar si la carpeta 'data' existe, y si no, crearla
    if not os.path.exists('data'):
        os.makedirs('data')

    # Guardar el DataFrame en un archivo de Excel en la carpeta 'data'
    save_path = save_results()
    df.to_excel(save_path, index=False)

## DEMORA 7 MINUTOS Y MEDIO EN CLASIFICAR 3233 DATOS
