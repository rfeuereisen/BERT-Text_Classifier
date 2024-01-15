import streamlit as st
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Load models
MODEL_CKPT_NIVEL_1 = Path("./models/dccuchile/distilbert-base-spanish-uncased-finetuned-nivel_1/")
MODEL_CKPT_NIVEL_2 = Path("./models/dccuchile/distilbert-base-spanish-uncased-finetuned-nivel_2/")
MODEL_CKPT_NIVEL_3 = Path("./models/dccuchile/distilbert-base-spanish-uncased-finetuned-nivel_3/")
MODEL_CKPT_COMENTARIO = Path("./models/dccuchile/distilbert-base-spanish-uncased-finetuned-comentario/")
MODEL_CKPT_NIVEL_TIPO_DETENCION = Path("./models/dccuchile/distilbert-base-spanish-uncased-finetuned-tipo_detencion/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_nivel_1 = AutoTokenizer.from_pretrained(MODEL_CKPT_NIVEL_1)
model_nivel_1 = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT_NIVEL_1).to(device)

tokenizer_nivel_2 = AutoTokenizer.from_pretrained(MODEL_CKPT_NIVEL_2)
model_nivel_2 = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT_NIVEL_2).to(device)

tokenizer_nivel_3 = AutoTokenizer.from_pretrained(MODEL_CKPT_NIVEL_3)
model_nivel_3 = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT_NIVEL_3).to(device)

tokenizer_comentario = AutoTokenizer.from_pretrained(MODEL_CKPT_COMENTARIO)
model_comentario = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT_COMENTARIO).to(device)

tokenizer_tipo_detencion = AutoTokenizer.from_pretrained(MODEL_CKPT_NIVEL_TIPO_DETENCION)
model_tipo_detencion = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT_NIVEL_TIPO_DETENCION).to(device)

def inference(text: str, model, tokenizer) -> str:
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]
    max_value = max(predictions)
    idx = predictions.index(max_value)
    return model.config.id2label[idx]

@st.cache(allow_output_mutation=True)
def load_data(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    return xls

def process_and_classify_data(df, selected_sheet):
    total_rows = df.shape[0]

    progress_bar = st.progress(0)

    for idx, row in enumerate(tqdm(df.iterrows(), total=total_rows, desc="Processing")):
        text = str(row[1]['Datos BI'])
        numero_reason = row[1]['Número Reason']

         # Condición 1: '3000'
        if str(df.loc[idx, 'Número Reason']) == '3000':
            category_comentario = inference(str(text), model_comentario, tokenizer_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
            category_nivel_1 = inference(str(text), model_nivel_1, tokenizer_nivel_1)
            if category_nivel_1 == "VACIO":
                category_nivel_1 = ""
            df.loc[idx, 'Nivel_1'] = category_nivel_1
            category_nivel_2 = inference(str(text), model_nivel_2, tokenizer_nivel_2)
            if category_nivel_2 == "VACIO":
                category_nivel_2 = ""
            df.loc[idx, 'Nivel_2'] = category_nivel_2
            category_nivel_3 = inference(str(text), model_nivel_3, tokenizer_nivel_3)
            if category_nivel_3 == "VACIO":
                category_nivel_3 = ""
            df.loc[idx, 'Nivel_3'] = category_nivel_3
            category_tipo = inference(str(text), model_tipo_detencion, tokenizer_tipo_detencion)
            if category_tipo == "VACIO":
                category_tipo = ""
            df.loc[idx, 'Tipo de Detención'] = category_tipo
        
         # Condición 2: '3010'
        if str(df.loc[idx, 'Número Reason']) == '3010':
            category_comentario = inference(str(text), model_comentario, tokenizer_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
            category_nivel_1 = inference(str(text), model_nivel_1, tokenizer_nivel_1)
            if category_nivel_1 == "VACIO":
                category_nivel_1 = ""
            df.loc[idx, 'Nivel_1'] = category_nivel_1
            category_nivel_2 = inference(str(text), model_nivel_2, tokenizer_nivel_2)
            if category_nivel_2 == "VACIO":
                category_nivel_2 = ""
            df.loc[idx, 'Nivel_2'] = category_nivel_2
            category_nivel_3 = inference(str(text), model_nivel_3, tokenizer_nivel_3)
            if category_nivel_3 == "VACIO":
                category_nivel_3 = ""
            df.loc[idx, 'Nivel_3'] = category_nivel_3
            category_tipo = inference(str(text), model_tipo_detencion, tokenizer_tipo_detencion)
            if category_tipo == "VACIO":
                category_tipo = ""
            df.loc[idx, 'Tipo de Detención'] = category_tipo

        # Condición 3: '3030'
        if str(df.loc[idx, 'Número Reason']) == '3030':
            category_comentario = inference(str(text), model_comentario, tokenizer_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
            df.loc[idx, 'Nivel_1'] = 'NEUMATICOS'
            df.loc[idx, 'Nivel_2'] = 'NEUMATICOS_Y_AROS'
            df.loc[idx, 'Nivel_3'] = None
            df.loc[idx, 'Tipo de Detención'] = "MPV"

        # Condición 4: '5000'
        if str(df.loc[idx, 'Número Reason']) == '5000':
            category_comentario = inference(str(text), model_comentario, tokenizer_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
            category_nivel_1 = inference(str(text), model_nivel_1, tokenizer_nivel_1)
            if category_nivel_1 == "VACIO":
                category_nivel_1 = ""
            df.loc[idx, 'Nivel_1'] = category_nivel_1
            category_nivel_2 = inference(str(text), model_nivel_2, tokenizer_nivel_2)
            if category_nivel_2 == "VACIO":
                category_nivel_2 = ""
            df.loc[idx, 'Nivel_2'] = category_nivel_2
            category_nivel_3 = inference(str(text), model_nivel_3, tokenizer_nivel_3)
            if category_nivel_3 == "VACIO":
                category_nivel_3 = ""
            df.loc[idx, 'Nivel_3'] = category_nivel_3
            category_tipo = inference(str(text), model_tipo_detencion, tokenizer_tipo_detencion)
            if category_tipo == "VACIO":
                category_tipo = ""
            df.loc[idx, 'Tipo de Detención'] = category_tipo

        # Condición 5: '5010'
        if str(df.loc[idx, 'Número Reason']) == '5010':
            category_comentario = inference(str(text), model_comentario, tokenizer_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
            category_nivel_1 = inference(str(text), model_nivel_1, tokenizer_nivel_1)
            if category_nivel_1 == "VACIO":
                category_nivel_1 = ""
            df.loc[idx, 'Nivel_1'] = category_nivel_1
            category_nivel_2 = inference(str(text), model_nivel_2, tokenizer_nivel_2)
            if category_nivel_2 == "VACIO":
                category_nivel_2 = ""
            df.loc[idx, 'Nivel_2'] = category_nivel_2
            category_nivel_3 = inference(str(text), model_nivel_3, tokenizer_nivel_3)
            if category_nivel_3 == "VACIO":
                category_nivel_3 = ""
            df.loc[idx, 'Nivel_3'] = category_nivel_3
            df.loc[idx, 'Tipo de Detención'] = 'MCA'

         # Condición 6: '5020'
        if str(df.loc[idx, 'Número Reason']) == '5020':
            category_comentario = inference(str(text), model_comentario, tokenizer_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
            category_nivel_1 = inference(str(text), model_nivel_1, tokenizer_nivel_1)
            if category_nivel_1 == "VACIO":
                category_nivel_1 = ""
            df.loc[idx, 'Nivel_1'] = category_nivel_1
            category_nivel_2 = inference(str(text), model_nivel_2, tokenizer_nivel_2)
            if category_nivel_2 == "VACIO":
                category_nivel_2 = ""
            df.loc[idx, 'Nivel_2'] = category_nivel_2
            category_nivel_3 = inference(str(text), model_nivel_3, tokenizer_nivel_3)
            if category_nivel_3 == "VACIO":
                category_nivel_3 = ""
            df.loc[idx, 'Nivel_3'] = category_nivel_3
            category_tipo = inference(str(text), model_tipo_detencion, tokenizer_tipo_detencion)
            if category_tipo == "VACIO":
                category_tipo = ""
            df.loc[idx, 'Tipo de Detención'] = category_tipo

        # Condición 7: '5030'
        if str(df.loc[idx, 'Número Reason']) == '5030':
            category_comentario = inference(str(text), model_comentario, tokenizer_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
            df.loc[idx, 'Nivel_1'] = 'NEUMATICOS_Y_AROS'
            df.loc[idx, 'Nivel_2'] = 'NEUMATICOS_Y_AROS'
            df.loc[idx, 'Nivel_3'] = None
            df.loc[idx, 'Tipo de Detención'] = 'MCV'

        # Condición 8: '5050'
        if str(df.loc[idx, 'Número Reason']) == '5050':
            category_comentario = inference(str(text), model_comentario, tokenizer_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
            category_nivel_1 = inference(str(text), model_nivel_1, tokenizer_nivel_1)
            if category_nivel_1 == "VACIO":
                category_nivel_1 = ""
            df.loc[idx, 'Nivel_1'] = category_nivel_1
            category_nivel_2 = inference(str(text), model_nivel_2, tokenizer_nivel_2)
            if category_nivel_2 == "VACIO":
                category_nivel_2 = ""
            df.loc[idx, 'Nivel_2'] = category_nivel_2
            category_nivel_3 = inference(str(text), model_nivel_3, tokenizer_nivel_3)
            if category_nivel_3 == "VACIO":
                category_nivel_3 = ""
            df.loc[idx, 'Nivel_3'] = category_nivel_3
            df.loc[idx, 'Tipo de Detención'] = 'MCM'

        # Condición 9: '5070'
        if str(df.loc[idx, 'Número Reason']) == '5070':
            category_comentario = inference(str(text), model_comentario, tokenizer_comentario)
            if category_comentario == "VACIO":
                category_comentario = ""
            df.loc[idx, 'Comentario'] = category_comentario
            category_nivel_1 = inference(str(text), model_nivel_1, tokenizer_nivel_1)
            if category_nivel_1 == "VACIO":
                category_nivel_1 = ""
            df.loc[idx, 'Nivel_1'] = category_nivel_1
            category_nivel_2 = inference(str(text), model_nivel_2, tokenizer_nivel_2)
            if category_nivel_2 == "VACIO":
                category_nivel_2 = ""
            df.loc[idx, 'Nivel_2'] = category_nivel_2
            category_nivel_3 = inference(str(text), model_nivel_3, tokenizer_nivel_3)
            if category_nivel_3 == "VACIO":
                category_nivel_3 = ""
            df.loc[idx, 'Nivel_3'] = category_nivel_3
            category_tipo = inference(str(text), model_tipo_detencion, tokenizer_tipo_detencion)
            if category_tipo == "VACIO":
                category_tipo = ""
            df.loc[idx, 'Tipo de Detención'] = category_tipo
        
        progress_bar.progress((idx + 1) / total_rows)

    return df

def main():
    st.title("Streamlit App for Text Classification")

    # File Upload
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        # Leer el archivo Excel
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names

        # Permitir al usuario seleccionar la hoja
        selected_sheet = st.selectbox("Select a sheet", sheet_names)

        # Cargar datos de la hoja seleccionada
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, engine='openpyxl')

        # Process and classify text
        df = process_and_classify_data(df, selected_sheet)  # Pasar selected_sheet

        # Display the processed DataFrame
        st.dataframe(df)

        # Save Button
        if st.button("Save Results"):
            # Save the DataFrame to Excel
            save_path = st.text_input("Enter the save path:", "output.xlsx")
            df.to_excel(save_path, index=False)
            st.success(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()
