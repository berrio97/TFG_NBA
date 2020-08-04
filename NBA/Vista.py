from tkinter import *
from tkinter.ttk import Separator
import numpy as np
from datetime import datetime

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkcalendar import Calendar
from sklearn.metrics import roc_curve, auc

import pandas as pd

from Controlador import Controlador

# Vista de la arquitectura Modelo-Vista-Controlador
class Vista(Frame):
    # Inicializa el controlador y algunas variables globales
    # Params:
    #   master: instancia de Window
    #   modelo: Modelo
    def __init__(self, master, modelo):
        Frame.__init__(self, master)
        self.pack()
        self.master = master
        self.modelo = modelo
        self.controlador = Controlador(self.modelo, self)
        self.col1 = 'khaki1'
        self.col2 = 'snow'
        self.font = 'Helvetica'
        self.init_components()
        return

    #Inicializa los componentes de la Vista y los coloca
    def init_components(self):
        self.label_model = Label(self.master, text='Modelo', font=(self.font, 14), relief=GROOVE, bg=self.col2)
        self.label_data = Label(self.master, text='Datos', font=(self.font, 14), relief=GROOVE, bg=self.col2)
        self.label_pred = Label(self.master, text='Predicción', font=(self.font, 14), relief=GROOVE, bg=self.col2)
        self.label_res = Label(self.master, text='Resultados', font=(self.font, 14), relief=GROOVE, bg=self.col2)

        self.variable = StringVar()
        self.variable.set('Modelo 1')
        self.variable.trace("w", self.change_model)

        self.model_selector = OptionMenu(self.master, self.variable, *list(self.modelo.modelos.keys()))

        self.frame = Frame(self.master)

        self.alg_label = Label(self.frame, text='Algoritmo', font=(self.font, 8, 'bold'), bg=self.col2,
                               relief='raised', width=20)
        self.seas_label = Label(self.frame, text='Temporadas', font=(self.font, 8, 'bold'), bg=self.col2,
                                relief='raised', width=20)
        self.columns_label = Label(self.frame, text='Selección columnas', font=(self.font, 8, 'bold'), bg=self.col2,
                                   relief='raised', width=20)
        self.pca_label = Label(self.frame, text='PCA', font=(self.font, 8, 'bold'), bg=self.col2,
                               relief='raised', width=20)
        self.params_label = Label(self.frame, text='Parámetros', bg=self.col2, font=(self.font, 8, 'bold'),
                                  relief='raised',
                                  width=20)

        self.alg_value = Label(self.frame, text='', font=(self.font, 8), bg=self.col2, relief='raised', width=30)
        self.seas_value = Label(self.frame, text='', font=(self.font, 8), bg=self.col2, relief='raised', width=30)
        self.columns_value = Label(self.frame, text='', font=(self.font, 8), bg=self.col2, relief='raised', width=30)
        self.pca_value = Label(self.frame, text='', font=(self.font, 8), bg=self.col2, relief='raised', width=30)
        self.params_value = Label(self.frame, text='', font=(self.font, 8), bg=self.col2, relief='raised', width=30)

        self.load_model_but = Button(self.master, text='Cargar modelo', state='disabled', command=self.load_model,
                                     bg=self.col2)

        self.train_model_but = Button(self.master, text='Entrenar modelo', state='disabled', command=self.train_model,
                                      bg=self.col2)

        self.load_data_but = Button(self.master, text='Cargar datos', command=self.load_data, bg=self.col2)

        self.ref_but = Button(self.master, text='Actualizar datos', command=self.refresh, bg=self.col2)

        self.home_label = Label(self.master, text='Equipo local', bg=self.col1)
        self.away_label = Label(self.master, text='Equipo visitante', bg=self.col1)

        self.home = StringVar()
        self.home.set(self.modelo.teams[0])
        self.homeOptionMenu = OptionMenu(self.master, self.home, *list(self.modelo.teams))
        self.homeOptionMenu.config(state='disabled')

        self.away = StringVar()
        self.away.set(self.modelo.teams[0])
        self.awayOptionMenu = OptionMenu(self.master, self.away, *list(self.modelo.teams))
        self.awayOptionMenu.config(state='disabled')

        self.calendar = Calendar(self.master, state='disabled')

        self.pred_but = Button(self.master, text='Hallar predicciones', state='disabled',
                               command=self.exec_prediction, bg=self.col2)

        self.result = Label(self.master, text='', bg=self.col1, font=(self.font, 10, 'bold'))
        self.pred = Label(self.master, text='', bg=self.col1, font=(self.font, 10, 'bold'))
        self.team_win = Label(self.master, text='', bg=self.col1, font=(self.font, 10, 'bold'))

        self.sep1 = Separator(self.master, orient=HORIZONTAL)
        self.sep2 = Separator(self.master, orient=HORIZONTAL)
        self.sep3 = Separator(self.master, orient=VERTICAL)

        self.label_error = Label(self.master, text='', font=('device', 10), fg='red', bg=self.col1)

        ### PACKING & PLACING
        self.label_model.pack()
        self.label_model.place(relx=0.05, rely=0.05, anchor=W)
        self.label_data.pack()
        self.label_data.place(relx=0.05, rely=0.4, anchor=W)
        self.label_pred.pack()
        self.label_pred.place(relx=0.05, rely=0.6, anchor=W)

        self.model_selector.pack()
        self.model_selector.place(relx=0.15, rely=0.15, anchor=CENTER)

        self.frame.pack()
        self.frame.place(relx=0.25, rely=0.05)

        self.alg_label.grid(row=0, rowspan=1, column=0, columnspan=1)
        self.seas_label.grid(row=1, rowspan=1, column=0, columnspan=1)
        self.columns_label.grid(row=2, rowspan=1, column=0, columnspan=1)
        self.pca_label.grid(row=3, rowspan=1, column=0, columnspan=1)
        self.params_label.grid(row=4, rowspan=1, column=0, columnspan=1, sticky=N + E + S + W)

        self.alg_value.grid(row=0, rowspan=1, column=1, columnspan=1)
        self.seas_value.grid(row=1, rowspan=1, column=1, columnspan=1)
        self.columns_value.grid(row=2, rowspan=1, column=1, columnspan=1)
        self.pca_value.grid(row=3, rowspan=1, column=1, columnspan=1)
        self.params_value.grid(row=4, rowspan=1, column=1, columnspan=1)
        self.change_model()

        self.load_model_but.pack()
        self.load_model_but.place(relx=0.1, rely=0.48, anchor=CENTER)

        self.train_model_but.pack()
        self.train_model_but.place(relx=0.24, rely=0.48, anchor=CENTER)

        self.load_data_but.pack()
        self.load_data_but.place(relx=0.38, rely=0.48, anchor=CENTER)

        self.ref_but.pack()
        self.ref_but.place(relx=0.52, rely=0.48, anchor=CENTER)

        self.home_label.pack()
        self.home_label.place(relx=0.1, rely=0.7, anchor=CENTER)

        self.away_label.pack()
        self.away_label.place(relx=0.25, rely=0.7, anchor=CENTER)

        self.homeOptionMenu.pack()
        self.homeOptionMenu.place(relx=0.1, rely=0.75, anchor=CENTER)

        self.awayOptionMenu.pack()
        self.awayOptionMenu.place(relx=0.25, rely=0.75, anchor=CENTER)

        self.calendar.pack()
        self.calendar.place(relx=0.45, rely=0.75, anchor=CENTER)

        self.pred_but.pack()
        self.pred_but.place(relx=0.17, rely=0.82, anchor=CENTER)

        self.label_res.pack()
        self.label_res.place(relx=0.7, rely=0.05, anchor=CENTER)

        self.result.pack()
        self.result.place(relx=0.8, rely=0.15, anchor=CENTER)

        self.pred.pack()
        self.pred.place(relx=0.8, rely=0.85, anchor=CENTER)

        self.team_win.pack()
        self.team_win.place(relx=0.8, rely=0.89, anchor=CENTER)

        self.sep1.place(relx=0.05, rely=0.33, relwidth=0.55)
        self.sep2.place(relx=0.05, rely=0.53, relwidth=0.55)
        self.sep3.place(relx=0.61, rely=0.05, relheight=0.9)

        self.label_error.place(relx=0.8, rely=0.93, anchor=CENTER)

    # Evento de cambiar el modelo que se esta seleccionando
    def change_model(self, *args):
        self.controlador.evento_change_model()
        if self.modelo.model_read is not None:
            self.alg_value['text'] = self.modelo.algorithms[self.modelo.model_read.alg]
            if self.modelo.model_read.seasons == '2015':
                seas = 'Desde 2014/2015'
            elif self.modelo.model_read.seasons == '2005':
                seas = 'Desde 2004/2005'
            else:
                seas = 'Desde 2000/2001'
            self.seas_value['text'] = seas
            self.columns_value['text'] = self.modelo.model_read.col
            self.pca_value['text'] = 'Sí' if self.modelo.model_read.pca_analysis else 'No'

            if not pd.isnull(self.modelo.model_read.params):
                aux = ''
                for key in list(eval(self.modelo.model_read.params).keys()):
                    aux += str(key) + ': ' + str(eval(self.modelo.model_read.params)[key]) + '\n'
                self.params_value['text'] = aux[:-1]
            else:
                self.params_value['text'] = ''

    # Evento de cargar los datos de los partidos
    def load_data(self):
        self.controlador.evento_load()
        if self.modelo.file is not None:
            self.load_model_but['state'] = 'active'
            self.calendar['state'] = 'normal'
            self.label_error.config(fg='green')
            self.label_error['text'] = 'Datos cargados con éxito'

    # Evento de actualizar los datos
    def refresh(self):
        self.controlador.evento_refresh()
        self.load_data()
        self.label_error.config(fg='green')
        self.label_error['text'] = 'Datos actualizados con éxito'

    # Evento de cargar el modelo predictivo seleccionado
    def load_model(self):
        self.label_error['text'] = ''
        self.train_model_but['state'] = 'active'
        self.controlador.evento_load_model()
        performance = self.modelo.modelo_prediccion.ac
        self.result['text'] = 'TASA DE ACIERTO: ' + str(performance.round(4) * 100) + '%'
        self.roc()
        self.pred_but['state'] = 'active'
        self.homeOptionMenu.config(state='active')
        self.awayOptionMenu.config(state='active')
        self.label_error.config(fg='green')
        self.label_error['text'] = 'Modelo cargado con éxito'

    # Evento de entrenar el modelo predictivo seleccionado
    def train_model(self):
        self.label_error['text'] = ''
        self.controlador.evento_train_model()
        self.load_model()

    # Evento de crear una curva ROC sobre los resultados del modelo predictivo seleccionado
    def roc(self):
        fpr, tpr, thres = roc_curve(self.modelo.modelo_prediccion.Y_test, self.modelo.modelo_prediccion.scores)
        auc_roc = auc(fpr, tpr)

        fig = Figure(figsize=(3.2, 3.2))
        a = fig.add_subplot(111)
        a.plot(fpr, tpr, color='blue', label='AUC %0.2f' % auc_roc)
        a.legend(loc="lower right")

        a.set_position([0.15, 0.12, 0.8, 0.8])
        a.set_xticks(ticks=np.arange(0, 1.5, 0.5))
        a.set_yticks(ticks=np.arange(0, 1.5, 0.5))
        a.set_xticklabels(labels=np.arange(0, 1.5, 0.5), fontdict={'fontsize': 8})
        a.set_yticklabels(labels=np.arange(0, 1.5, 0.5), fontdict={'fontsize': 8})
        a.set_title("Curva ROC " + self.modelo.algorithms[self.modelo.model_read.alg], fontsize=10)
        a.set_ylabel("TPR", fontsize=8)
        a.set_xlabel("FPR", fontsize=8)

        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.get_tk_widget().pack(expand=True)
        canvas.get_tk_widget().place(relx=0.8, rely=0.5, anchor=CENTER)
        canvas.draw()

    # Evento de crear las predicciones para un partido determinado
    def exec_prediction(self):
        date = datetime.strptime(self.calendar.get_date(), '%m/%d/%y')
        game_id = str(date.year) + '%02d' % date.month + '%02d' % date.day + '0' + self.home.get()

        if self.home.get() != self.away.get() and self.home.get() != 'No seleccionado' and self.away.get() != 'No seleccionado':
            aux = self.modelo.modelo_prediccion.predictions_test
            game_true = game_id in aux.game_id.values
            if date < datetime.today() and game_true and aux[aux.game_id == game_id]['a_team_id'].values[
                0] == self.away.get() or date > datetime.today():
                self.controlador.evento_exec_prediction()
                predres = self.modelo.prediction
                aux = self.home.get() if predres else self.away.get()
                self.label_error['text'] = ''
                self.pred['text'] = str(self.home.get()) + ': ' + str(predres.round(2)) + '\t' + str(
                    self.away.get()) + ': ' + str((1 - predres).round(2))
                if date<datetime.today():
                    self.team_win['text'] = 'Victoria real: ' + str(aux)
            else:
                self.label_error.config(fg='red')
                self.label_error['text'] = 'ERROR: Ese partido no se ha disputado.'
                self.pred['text'] = ''
                self.team_win['text'] = ''
        elif self.home.get() == 'No seleccionado' and self.away.get() == 'No seleccionado':
            self.label_error.config(fg='red')
            self.pred['text'] = ''
            self.team_win['text'] = ''
            self.label_error['text'] = 'ERROR: Hay que determinar los equipos'
        elif self.home.get() == 'No seleccionado' or self.away.get() == 'No seleccionado':
            self.label_error.config(fg='red')
            self.pred['text'] = ''
            self.team_win['text'] = ''
            self.label_error['text'] = 'ERROR: Falta un equipo por determinar'
        elif self.home.get() == self.away.get():
            self.label_error.config(fg='red')
            self.pred['text'] = ''
            self.team_win['text'] = ''
            self.label_error['text'] = 'ERROR: Los equipos deben ser diferentes.'
