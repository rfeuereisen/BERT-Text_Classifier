from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import regex as re

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from sklearn.model_selection import train_test_split

CATEGORY = "nivel_3" #tipo_detencion, nivel_1, nivel_2, nivel_3, comentario
RAW_DATASET = f"{CATEGORY}.csv"

def preprocess_comment(comment):
    comment = re.sub(r'-\d+', '', comment)  # Eliminar valores negativos como -1, -2, etc.
    comment = comment.lower()
    comment = re.sub(r'\bch\b', 'chequeo', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\bcheq\b', 'chequeo', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\btemp\b', 'temperatura', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\btemperatu\b', 'temperatura', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\btem\b', 'temperatura', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\brefrigerant\b', 'refrigerante', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\brefrig\b', 'refrigerante', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\bpres\b', 'presion', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\bvolt\b', 'voltaje', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\bcod\b', 'codigo', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\beval\b', 'evaluacion', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\bpinch\b', 'pinchado', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\bpro\b', 'problema', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\bprob\b', 'problema', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\bprobl\b', 'problema', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\brell\b', 'relleno', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\brep\b', 'reparacion', comment, flags=re.IGNORECASE)
    comment = re.sub(r'\bpm\.?\b', 'mantencion programada', comment, flags=re.IGNORECASE)
    return comment

if __name__ == "__main__":
    raw_data = pd.read_csv(Path("data") / RAW_DATASET, encoding="ISO-8859-1")
    raw_data=raw_data.applymap(str)
    raw_data['label'] = raw_data['label'].apply(lambda x: x.strip())
    raw_data['label'] = raw_data['label'].replace('nan', "VACIO")
    raw_data['text'] = raw_data['text'].apply(preprocess_comment)
    raw_data.dropna(subset="text", inplace=True)
    raw_data.dropna(subset="label", inplace=True)
    df_cleaned = raw_data.drop_duplicates()

    df_labels = df_cleaned['label'].value_counts().rename_axis('unique_values').reset_index(name='counts')
    df_labels_top_n = df_labels.loc[df_labels['counts'] > 200]
    categories = list(df_labels_top_n.unique_values)
    dict_categories = dict(enumerate(categories))
    dict_categories_swap = {v: k for k, v in dict_categories.items()}
    json_object = json.dumps(dict_categories, indent = 4)
    with open(f"data/{CATEGORY}_categories.json", "w") as outfile: 
        json.dump(dict_categories, outfile)

    filtered_df = df_cleaned[df_cleaned["label"].isin(categories)]
    filtered_df = filtered_df.replace({'label': dict_categories_swap})

    train, test = train_test_split(filtered_df, test_size=0.2)
    train.to_csv(f'data/train_{CATEGORY}_filtered.csv', index=False)
    test.to_csv(f'data/test_{CATEGORY}_filtered.csv', index=False)







