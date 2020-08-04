from tkinter import Tk

# Controlador de la arquitectura Modelo-Vista-Controlador
class Controlador:
    # Se inicializa el Controlador y se asocia con el Modelo y la Vista ya existentes
    # Params:
    #   modelo: Modelo
    #   vista: Vista
    def __init__(self,  modelo, vista):
        self.modelo = modelo
        self.vista = vista

    # Manjeador del evento de cambiar el modelo predictivo
    def evento_change_model(self):
        self.modelo.set_modelo_prediccion(self.vista.variable.get())

    # Manejador del evento de cargar los datos
    def evento_load(self):
        self.modelo.load_file()

    # Manejador del evento de actualizar los datos
    def evento_refresh(self):
        self.modelo.refresh_files()

    # Manejador del evento de cargar el modelo predictivo
    def evento_load_model(self):
        self.modelo.load_model()

    # Manejador del evento de entrenar el modelo predictivo
    def evento_train_model(self):
        self.modelo.train_model()

    # Manejador del evento de hallar las predicciones
    def evento_exec_prediction(self):
        h_team = self.vista.home.get()
        a_team = self.vista.away.get()
        date = self.vista.calendar.get_date()
        self.modelo.execute_predict(h_team, a_team, date)

