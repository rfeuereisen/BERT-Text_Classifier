import tkinter as tk
from tkinter import simpledialog
import pandas as pd

# Crea una ventana principal
root = tk.Tk()
root.withdraw()  # Oculta la ventana principal

# Carga tus datos desde el archivo Excel
df = pd.read_excel('data/Resultados.xlsx')

# Lista para almacenar clasificaciones incorrectas
clasificaciones_incorrectas = []

# Bucle para revisar las clasificaciones incorrectas
while True:
    respuesta = simpledialog.askstring("Clasificación incorrecta", "¿Hubo alguna clasificación incorrecta? (Sí/No)")
    
    if respuesta.lower() != 'si':
        break  # Sal del bucle si la respuesta no es 'Sí'
    
    dato_incorrecto = simpledialog.askstring("Dato incorrecto", "¿Qué dato fue mal clasificado?")
    
    # Pregunta por la clasificación correcta para cada categoría
    clasificacion_correcta = {}
    for categoria in ["Comentario", "Nivel_1", "Nivel_2", "Nivel_3", "Tipo de Detención"]:
        clasificacion = simpledialog.askstring("Clasificación correcta", f"¿Cuál es la clasificación correcta para {categoria}?")
        clasificacion_correcta[categoria] = clasificacion
    
    # Almacena la información en la lista de clasificaciones incorrectas
    clasificaciones_incorrectas.append({
        "Dato incorrecto": dato_incorrecto,
        "Clasificación correcta": clasificacion_correcta
    })

# Guarda las clasificaciones incorrectas en un archivo Excel
if len(clasificaciones_incorrectas) > 0:
    feedback_df = pd.DataFrame(clasificaciones_incorrectas)
    feedback_df.to_excel('data/feedback.xlsx', index=False)
    print("Clasificaciones incorrectas guardadas en feedback.xlsx")

# Cierra la ventana emergente
root.destroy()
