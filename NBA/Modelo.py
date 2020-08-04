import pandas as pd
from GestorBBDD import GestorBBDD
import pickle
from ModeloPrediccion import ModeloPrediccion
from datetime import datetime

# Modelo de la arquitectura Modelo-Vista-Controlador
class Modelo:
    # Inicializa el Modelo y crea algunas variables globales del mismo
    def __init__(self):
        self.modelFile = pd.read_csv('modelos/models.csv')
        self.modelos = {'Modelo 1': 0, 'Modelo 2': 1, 'Modelo 3': 2, 'Modelo 4': 3}
        self.switchcol = {'all': 'all',
                          'corr': ['h_pytha', 'h_streak', 'h_win_pct_total', 'h_win_pct_home',
                                   'a_pytha', 'a_streak', 'a_win_pct_total', 'a_win_pct_away',
                                   'head_to_head'],
                          'manual': ['h_proj_win', 'h_win_pct_total', 'h_win_pct_home', 'h_ort',
                                     'h_drt', 'h_eFG', 'h_streak', 'h_pace',
                                     'a_proj_win', 'a_win_pct_total', 'a_win_pct_away', 'a_ort',
                                     'a_drt', 'a_eFG', 'a_streak', 'a_pace']}
        self.teams = ['No seleccionado', 'ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU',
                      'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR',
                      'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
        self.algorithms = {'lr': 'Regresión Lineal', 'lg': 'Regresión Logística', 'svm': 'SVM',
                           'mlp': 'Perceptrón Multicapa', 'forest': 'Random Forest', 'knn': 'K-Vecinos'}
        self.file = None
        self.model_read = None
        self.modelo_prediccion = None
        self.execution = None
        self.col_exe = None
        self.gestor = GestorBBDD('../datos/')
        self.tasa_acierto = None
        self.scores = None
        self.test = None
        self.prediction = None
        self.name = list(self.modelos.keys())[0]

    # Cambia el modelo predictivo seleccionado en la Vista
    def set_modelo_prediccion(self, index):
        self.model_read = self.modelFile.iloc[self.modelos[index], :]
        self.name = index

    # Carga los datos de los partidos
    def load_file(self):
        self.file = self.gestor.get_games()

    # Carga el modelo predictivo seleccionado
    def load_model(self):
        aux = self.name + '.pkl'
        with open('modelos/'+aux, 'rb') as f:
            self.modelo_prediccion = pickle.load(f)
            f.close()
    # Entrena el modelo predictivo seleccionado
    def train_model(self):
        pca = True if self.modelo_prediccion.pca is not False else False
        params = eval(self.model_read.params) if not pd.isnull(self.model_read.params) else None
        m = ModeloPrediccion(self.file, seas=self.modelo_prediccion.seasons, columns=self.modelo_prediccion.col,
                             pca_analysis=pca,
                             wrapper=self.modelo_prediccion.wrapper, standarize=self.modelo_prediccion.standarize)
        m.execute(self.model_read.alg, params)
        self.modelo_prediccion = m
        with open('modelos/'+self.name+'.pkl', 'wb') as f:
            pickle.dump(self.modelo_prediccion, f, -1)
            f.close()

    # Actualiza los datos utilizados
    def refresh_files(self):
        self.gestor.refresh()

    # Halla las predicciones para un partido determinado por el usuario
    # Params:
    #   h_team: identificador equipo local
    #   a_team: identificador equipo visitante
    #   date: fecha del partido
    def execute_predict(self, h_team, a_team, date):
        date = datetime.strptime(date, '%m/%d/%y')
        if date < datetime.today():
            game_id = str(date.year) + '%02d' % date.month + '%02d' % date.day + '0' + h_team
            aux = self.modelo_prediccion.predictions_test[self.modelo_prediccion.predictions_test.game_id == game_id]
            self.prediction = aux[0].values[0]
            self.victory = aux['victory'].values[0]
        else:
            pred = pd.DataFrame({'h_team_id': [h_team], 'a_team_id': [a_team], 'date': [date]})
            self.prediction = self.modelo_prediccion.pred_function(pred)
