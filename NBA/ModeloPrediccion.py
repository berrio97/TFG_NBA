import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from GestorBBDD import GestorBBDD
import matplotlib.pyplot as plt
import pickle

pd.set_option('display.max_columns', None)

# Crea los modelos predictivos sobre los datos de los partidos
class ModeloPrediccion:
    # Inicializa el modelo predictivo y crea las variables globales que se van a usar
    # Params:
    #   db: archivo con los partidos
    #   columns: atributos a utilizar
    #   seas: temporadas a usar
    #   pca_analysis: realizar pca o no
    #   wrapper: wrapper o no
    #   cv: folds de la validacion cruzada
    #   tune: realizar la validacion cruzada o no
    def __init__(self, db, columns='all', seas='all', pca_analysis=False, wrapper=False, cv=10, tune=False,
                 standarize='normalize'):
        self.fixed_columns = ['game_id', 'h_team_id', 'a_team_id', 'season', 'date']
        db['date'] = pd.to_datetime(db['date'])
        current_season = np.unique(db.season)[-1]
        # Selecciona las columnas
        if columns == 'all':
            self.col = list(set(db.columns) - set(self.fixed_columns))
            self.col.remove('victory')
        else:
            self.col = columns
        # Selecciona las temporadas, si es un array contiene las temporadas a usar como entrenamiento, si es un solo
        # entero, se refiere a los periodos de quince dias de una misma temporada
        if isinstance(seas, str) or isinstance(seas, list) or isinstance(seas, np.ndarray):
            if seas != 'all':
                seasons = list(seas) + [current_season]
                db = db.loc[db.apply(lambda row: row.season in seasons, axis=1), :]
            aux = db.loc[
                ((db['date'].map(lambda x: x.month) < 8) | (db['date'].map(lambda x: x.month) > 10))].reset_index(
                drop=True)
            aux = aux.fillna(0)
            XV = aux[aux.season != current_season].reset_index(drop=True)

            test_set = aux[aux.season == current_season].reset_index(drop=True)

        elif isinstance(seas, int):
            aux = db[db.season == (current_season)].reset_index(drop=True)
            aux = aux.dropna()
            first_date = sorted(aux['date'].values)[0]
            XV = aux[(aux.date >= first_date) & (
                    aux.date <= first_date + np.timedelta64(15 * (seas), 'D'))].reset_index(drop=True)

            test_set = aux[(aux.date > first_date + np.timedelta64(15 * (seas), 'D')) & (
                    aux.date <= first_date + np.timedelta64(15 * (seas + 1), 'D'))].reset_index(drop=True)

        self.X_XV = XV[self.fixed_columns + self.col]
        self.Y_XV = XV.victory

        self.X_test = test_set[self.fixed_columns + self.col]
        self.Y_test = test_set.victory

        # Estandariza los datos
        if standarize is not None:
            if standarize == 'normalize':
                self.scaler = preprocessing.StandardScaler().fit(self.X_XV[self.col])
                self.X_test[self.col] = preprocessing.StandardScaler().fit(self.X_test[self.col]).transform(
                    self.X_test[self.col])
                self.X_XV[self.col] = self.scaler.transform(self.X_XV[self.col])
            elif standarize == 'scale':
                self.scaler = preprocessing.MinMaxScaler().fit(self.X_XV[self.col])
                self.X_test[self.col] = preprocessing.MinMaxScaler().fit(self.X_test[self.col]).transform(
                    self.X_test[self.col])
                self.X_XV[self.col] = self.scaler.transform(self.X_XV[self.col])
        else:
            self.scaler = None
        self.standarize=standarize
        self.pca = pca_analysis
        self.tune = tune
        self.cv = cv
        self.params = None
        self.wrapper = wrapper
        self.algorithms = {'lr': self.funLinReg, 'lg': self.funLogReg, 'svm': self.funSVM, 'mlp': self.funMLP,
                           'forest': self.funRand, 'knn': self.funKNN, 'bayes': self.funBayes}
        self.algorithms_tune = {'lr': LinearRegression(), 'lg': LogisticRegression(max_iter=200),
                                'svm': svm.SVC(probability=True),
                                'mlp': MLPClassifier(),
                                'forest': RandomForestClassifier(max_features='log2'),
                                'knn': KNeighborsClassifier(),
                                'bayes': MultinomialNB()}

        self.seasons = seas
        self.execution = None
        self.scores = None
        self.predictions_test = None
        self.ac = None

    # Da el formato deseado al conjunto de datos con los partidos que se quieren predecir
    # Params:
    #   pred: dataframe con identificador del equipo local y visitante y fecha del partido
    def format_pred(self, pred):
        current = GestorBBDD('C:/Users/PC/Dropbox/5Indat/2o Cuatri/TFG Informatica/files/').get_current()
        current['date'] = pd.to_datetime(current['date'])
        teams = [x + 'g' for x in np.unique(current.team_id)] + [y + 'w' for y in np.unique(current.team_id)]
        pred['date'] = pd.to_datetime(pred['date'])
        game_id = pred.apply(lambda row: row.date.strftime('%Y') + row.date.strftime('%m') + row.date.strftime(
            '%d') + '0' + row.h_team_id, axis=1)

        home = pd.DataFrame(columns=current.columns)
        away = pd.DataFrame(columns=current.columns)
        for iter, row in pred.iterrows():
            home = home.append(current[current.team_id == row['h_team_id']]).reset_index(drop=True)
            home.loc[iter, 'head_to_head'] = (current[current.team_id == row.h_team_id][row.a_team_id + 'w'] /
                                              current[current.team_id == row.h_team_id][row.a_team_id + 'g']).values
            home.loc[iter, 'rest'] = (pred.loc[iter, 'date'] - home.loc[iter, 'date']).days

            away = away.append(current[current.team_id == row['a_team_id']]).reset_index(drop=True)
            away.loc[iter, 'rest'] = (pred.loc[iter, 'date'] - away.loc[iter, 'date']).days
        home['game_id'] = game_id
        home['date'] = pred['date']
        home = home.drop(teams, axis=1)
        away['game_id'] = game_id
        away['date'] = pred['date']
        away = away.drop(teams, axis=1)

        t = GestorBBDD()
        joined = t.join_ha(home, away)
        pred_X = joined.drop('victory', axis=1)
        return pred_X

    # Entrena el modelo creado a partir del algoritmo y sus parametros, y halla la tasa de acierto sobre
    # el conjunto de test
    # Params:
    #   alg: tipo de tecnica de aprendizaje
    #   params: parametros del clasificador
    # Devuelve la tasa de acierto sobre el conjunto de test
    def execute(self, alg, params=None):
        col = self.col
        # Seleccion de atributos mediante wrapper
        if self.wrapper:
            aux = self.algorithms[alg]()
            if alg == 'lr' or alg == 'lg':
                rfe = RFECV(aux, cv=10)
            else:
                rfe = RFE(aux, n_features_to_select=10)
            rfe = rfe.fit(self.X_XV.drop(self.fixed_columns, 1), self.Y_XV)
            col = list(self.X_XV.drop(self.fixed_columns, 1).columns[rfe.support_])
            self.X_XV = self.X_XV[self.fixed_columns + col]
            self.X_test = self.X_test[self.fixed_columns + col]

        # Reduccion del espacio de caracteristicas mediante PCA
        if self.pca:
            self.pca = PCA(n_components=0.7, svd_solver='full').fit(self.X_XV[col])
            auxXV = self.pca.transform(self.X_XV[col])
            self.X_XV = pd.concat([self.X_XV[self.fixed_columns], pd.DataFrame(auxXV)], axis=1)
            auxTest = self.pca.transform(self.X_test[col])
            self.X_test = pd.concat([self.X_test[self.fixed_columns], pd.DataFrame(auxTest)], axis=1)

        # Realiza la validacion cruzada o una ejecucion estandar
        if not self.tune or params is None:
            self.params = params
            algorithm = self.algorithms[alg]()
        else:
            grid = GridSearchCV(self.algorithms_tune[alg], params, refit=True, verbose=1, cv=self.cv, n_jobs=3)
            grid.fit(self.X_XV.drop(self.fixed_columns, axis=1), self.Y_XV)
            print(grid.best_estimator_)
            print('\tAcierto entrenamiento: ' + str(grid.best_score_))
            algorithm = grid.best_estimator_

        self.execution = algorithm.fit(self.X_XV.drop(self.fixed_columns, axis=1), self.Y_XV)
        if alg == 'lr':
            self.scores = self.execution.predict(self.X_test.drop(self.fixed_columns, axis=1))
        else:
            self.scores = self.execution.predict_proba(self.X_test.drop(self.fixed_columns, axis=1))[:, 1]
        aux = self.scores > 0.5
        self.predictions_test = pd.concat([self.X_test[self.fixed_columns], pd.DataFrame(self.scores),self.Y_test], axis=1)
        self.ac = accuracy_score(self.Y_test, aux)

        print('Acierto ' + str(alg) + ' ' + str(self.ac))
        return self.ac

    # Predice el resultado de un partido aportado por el usuario
    # Params:
    #   pred: dataframe con identificador del equipo local y visitante y fecha del partido
    def pred_function(self, pred):
        if not pred.empty:
            X = self.format_pred(pred)
            X = X[self.fixed_columns + self.col]
            if self.scaler is not None:
                X[self.col] = self.scaler.transform(X[self.col])
        if self.pca is not False:
            auxPred = self.pca.transform(X[self.col])
            X = pd.concat([X[self.fixed_columns], pd.DataFrame(auxPred)], axis=1).fillna(0)
        X = X.fillna(0)
        if isinstance(self.execution, LinearRegression):
            prediction = self.execution.predict(X.drop(self.fixed_columns, axis=1))[0]
        else:
            prediction = self.execution.predict_proba(X.drop(self.fixed_columns, axis=1))[0][1]
        print('\tPrediccion ' + str(prediction))
        return prediction

    # Devuelve una instancia de regresion lineal
    def funLinReg(self):
        return LinearRegression()

    # Devuelve una instancia de regresion logistica
    def funLogReg(self):
        if self.params is not None:
            return LogisticRegression(C=self.params['C'], solver=self.params['solver'], max_iter=200)
        else:
            return LogisticRegression(max_iter=200)

    # Devuelve una instancia de SVM
    def funSVM(self):
        if self.params is not None:
            return svm.SVC(C=self.params['C'], kernel=self.params['kernel'], probability=True)
        else:
            return svm.SVC(C=1, kernel='linear', probability=True)

    # Devuelve una instancia de perceptron multicapa
    def funMLP(self):
        if self.params is not None:
            return MLPClassifier(hidden_layer_sizes=self.params['hidden_layer_sizes'], alpha=self.params['alpha'],
                                 activation=self.params['activation'], solver=self.params['solver'])
        else:
            return MLPClassifier(hidden_layer_sizes=(5, 2), alpha=0.01, activation='logistic', solver='sgd')

    # Devuelve una instancia del random forest
    def funRand(self):
        if self.params is not None:
            return RandomForestClassifier(n_estimators=self.params['n_estimators'], criterion=self.params['criterion'],
                                          max_features='log2')
        else:
            return RandomForestClassifier(max_features='log2')

    # Devuelve una instancia de k-vecinos
    def funKNN(self):
        if self.params is not None:
            return KNeighborsClassifier(n_neighbors=self.params['n_neighbors'])
        else:
            return KNeighborsClassifier(n_neighbors=36)

    # Devuelve una instancia de naive bayes
    def funBayes(self):
        if self.params is not None:
            return MultinomialNB(alpha=self.params['alpha'], fit_prior=self.params['fit_prior'])
        else:
            return MultinomialNB(alpha=10)