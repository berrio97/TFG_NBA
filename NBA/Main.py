from tkinter import Tk

from Vista import Vista
from ModeloPrediccion import ModeloPrediccion
from Modelo import Modelo

# Inicia el modelo y la vista de la herramienta
class Main:
    # Inicializa la clase Main y crea las variables necesarias para la interfaz grafica
    def __init__(self):
        self.modelo = Modelo()
        window = Tk()
        window.configure(bg='khaki1')
        window.geometry('1100x550')
        window.resizable(width=False, height=False)
        window.title('Analisis NBA')
        self.vista = Vista(window, self.modelo)
        self.vista.mainloop()
    # Devuelve el Modelo
    def get_modelo(self):
        return self.modelo

if __name__ == '__main__':
    Main()
