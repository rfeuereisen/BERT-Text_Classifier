import tkinter as tk
from tkinter import simpledialog
import pandas as pd

# Función para mostrar la ventana emergente de clasificación correcta
def mostrar_ventana_clasificacion_correcta():
    # Crear una ventana emergente para ingresar las clasificaciones correctas
    clasificacion_correcta_dialog = tk.Toplevel(root)
    clasificacion_correcta_dialog.title("Clasificación correcta")

    # Etiquetas y campos de entrada para cada categoría
    categorias = ["Comentario", "Nivel_1", "Nivel_2", "Nivel_3", "Tipo de Detención"]
    clasificacion_correcta_entries = {}

    for categoria in categorias:
        tk.Label(clasificacion_correcta_dialog, text=f"{categoria}:").pack()
        entry = tk.Entry(clasificacion_correcta_dialog)
        entry.pack()
        clasificacion_correcta_entries[categoria] = entry

    # Botón para confirmar las clasificaciones correctas
    confirmar_button = tk.Button(clasificacion_correcta_dialog, text="Confirmar", command=clasificacion_correcta_dialog.destroy)
    confirmar_button.pack()

    return clasificacion_correcta_entries, clasificacion_correcta_dialog

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

    # Mostrar la ventana emergente de clasificación correcta
    clasificacion_correcta_entries, clasificacion_correcta_dialog = mostrar_ventana_clasificacion_correcta()

    # Obtener las clasificaciones correctas ingresadas por el usuario
    clasificacion_correcta = {categoria: entry.get() for categoria, entry in clasificacion_correcta_entries.items()}

    # Almacena la información en la lista de clasificaciones incorrectas
    clasificaciones_incorrectas.append({
        "Dato incorrecto": dato_incorrecto,
        "Clasificación correcta": clasificacion_correcta
    })

    # Cerrar la ventana de clasificación correcta
    clasificacion_correcta_dialog.wait_window()

# Guarda las clasificaciones incorrectas en un archivo Excel
if len(clasificaciones_incorrectas) > 0:
    feedback_df = pd.DataFrame(clasificaciones_incorrectas)
    feedback_df.to_excel('data/feedback.xlsx', index=False)
    print("Clasificaciones incorrectas guardadas en feedback.xlsx")

# Cierra la ventana principal
root.destroy()
