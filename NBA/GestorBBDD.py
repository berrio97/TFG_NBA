import pandas as pd
import numpy as np
from datetime import datetime
import timeit

# Gestiona los conjuntos de datos con los que se trabaja durante el proyecto
class GestorBBDD:
    # Inicializa el gestor y las variables propias del mismo
    # Params:
    #   path: ruta donde se encuentran los archivos y donde se quieren guardar los creados
    def __init__(self, path=None):
        # Variables propias del gestor, las columnas son las propias de un box score tradicional y las fechas
        # son en las que comienzan los playoffs, para poder filtrar
        self.path = path
        self.columns = ['ast', 'blk', 'drb', 'fg', 'fg3', 'fg3a', 'fga', 'ft', 'fta',
                        'mp', 'orb', 'pf', 'pts', 'stl', 'tov', 'trb']
        self.dates = {2001: datetime(2001, 4, 21), 2002: datetime(2002, 4, 20),
                      2003: datetime(2003, 4, 19), 2004: datetime(2004, 4, 17),
                      2005: datetime(2005, 4, 23), 2006: datetime(2006, 4, 22),
                      2007: datetime(2007, 4, 21), 2008: datetime(2008, 4, 19),
                      2009: datetime(2009, 4, 18), 2010: datetime(2010, 4, 17),
                      2011: datetime(2011, 4, 16), 2012: datetime(2012, 4, 28),
                      2013: datetime(2013, 4, 20), 2014: datetime(2014, 4, 19),
                      2015: datetime(2015, 4, 18), 2016: datetime(2016, 4, 16),
                      2017: datetime(2017, 4, 15), 2018: datetime(2018, 4, 14),
                      2019: datetime(2019, 4, 13), 2020: datetime(2020, 4, 20)}
        self.bef_columns = self.columns + ['pt_diff', 'poss', 'pt_allowed', 'drb_opp', 'orb_opp']
        self.franchises = []

    # A partir del conjunto de datos Raw, se obtienen los box score tradicionales por jugador y partido
    # Params:
    #   database: dataframe con los datos en Raw
    # Puede devolver un dataframe con los box score
    def create_traditional_boxscore(self, database, ret=False):
        boxscore_df = database[database['box_type'] == 'game-basic']
        boxscore_df = boxscore_df.drop(['box_type'], axis=1)
        boxscore_df = boxscore_df.fillna(0)
        boxscore_df = boxscore_df.apply(pd.to_numeric, errors='coerce').fillna(boxscore_df)
        boxscore_df['date'] = pd.to_datetime(boxscore_df['date'])
        boxscore_df = boxscore_df.rename(columns={'ishome': 'home'})
        boxscore_df = boxscore_df.groupby(['game_id', 'team_id', 'player_id']).first()[
            self.columns + ['sp', 'date', 'home']].reset_index()
        if ret:
            return boxscore_df
        else:
            boxscore_df.to_csv(self.path + 'boxscore.csv', index=False)

    # A partir de los datos de los box score se consiguen datos para el total de cada equipo por partido
    # Params:
    #   boxscore: dataframe con los box scores
    # Puede devolver un dataframe con las estadisticas de los box score por equipo
    def create_team_totals(self, boxscore, ret=False):
        team_boxscore = boxscore.copy()
        team_boxscore = team_boxscore.groupby(['game_id', 'team_id', 'date', 'home'], as_index=False).sum().reset_index(
            drop=True)
        team_boxscore['mp'] = team_boxscore['sp'] / 60
        team_boxscore['season'] = team_boxscore['date'].apply(
            lambda row: row.year if row.month <= 8 else row.year + 1)
        team_boxscore = team_boxscore.loc[
                        team_boxscore.apply(lambda row: row['date'] < self.dates[row['season']], axis=1), :]
        team_boxscore = team_boxscore.groupby(['season', 'team_id', 'game_id'], as_index=False).first()

        aux = self.get_franchises()
        franchises_dict = dict(zip(aux.team_id, aux.franchise))
        team_boxscore['franchise'] = team_boxscore['team_id'].apply(lambda x: franchises_dict[x])
        team_boxscore['aux'] = team_boxscore.apply(lambda row: (
            team_boxscore.loc[
                (team_boxscore['game_id'] == row['game_id']) & (team_boxscore['team_id'] != row['team_id'])][
                ['pts', 'drb', 'fga', 'fta', 'orb', 'fg', 'tov', 'team_id', 'franchise']]).values[0], axis=1)
        team_boxscore[
            ['pt_allowed', 'drb_opp', 'fga_opp', 'fta_opp', 'orb_opp', 'fg_opp', 'tov_opp', 'opp',
             'fr_opp']] = pd.DataFrame(
            team_boxscore.aux.tolist(), index=team_boxscore.index)

        team_boxscore['victory'] = team_boxscore['pts'] > team_boxscore['pt_allowed']
        team_boxscore['pt_diff'] = team_boxscore['pts'] - team_boxscore['pt_allowed']
        team_boxscore['poss'] = 0.5 * (
                (team_boxscore['fga'] + 0.4 * team_boxscore['fta'] - 1.07 * (
                        team_boxscore['orb'] / (team_boxscore['orb'] + team_boxscore['drb_opp'])) * (
                         team_boxscore['fga'] - team_boxscore['fg']) + team_boxscore['tov']) + (
                        team_boxscore['fga_opp'] + 0.4 * team_boxscore['fta_opp'] - 1.07 * (
                        team_boxscore['orb_opp'] / (team_boxscore['orb_opp'] + team_boxscore['drb'])) * (
                                team_boxscore['fga_opp'] - team_boxscore['fg_opp']) + team_boxscore['tov_opp']))

        self.franchises = np.unique(team_boxscore['franchise'])
        team_boxscore[self.franchises + 'g'] = pd.DataFrame([[0] * len(self.franchises)] * team_boxscore.shape[0])
        team_boxscore[self.franchises + 'w'] = pd.DataFrame([[0] * len(self.franchises)] * team_boxscore.shape[0])
        for i in self.franchises:
            team_boxscore[i + 'g'] = team_boxscore.apply(
                lambda row: 1 if row.fr_opp == i else 0, axis=1)
            team_boxscore[i + 'w'] = team_boxscore.apply(
                lambda row: row.victory if row.fr_opp == i else False, axis=1)

        teams = [x + 'g' for x in self.franchises] + [y + 'w' for y in self.franchises]
        team_boxscore[teams] = team_boxscore[teams].astype(int)
        team_boxscore = team_boxscore.drop(['aux', 'fga_opp', 'fta_opp', 'fg_opp', 'tov_opp', 'sp'],
                                           axis=1).reset_index(drop=True)
        if ret:
            return team_boxscore
        else:
            team_boxscore.to_csv(self.path + 'team_totals.csv', index=False)

    # Se consiguen las estadisticas previas a un partido para conocer la situación de cada equipo y calcular
    # estadísticas avanzadas
    # Params:
    #   score_by_team: dataframe con las estadisticas de un boxscore por equipo
    #   streak: numero de partidos a tener en cuenta para la racha
    def create_before(self, score_by_team, streak=5):
        previous_bs = score_by_team.groupby(['season', 'team_id', 'game_id'], as_index=False).first()
        self.franchises = np.unique(previous_bs.franchise)
        teams = [x + 'g' for x in self.franchises] + [y + 'w' for y in self.franchises]
        previous_bs[teams] = previous_bs[teams].fillna(0)
        previous_bs[self.franchises + 'g'] = previous_bs.groupby(['franchise'])[self.franchises + 'g'].cumsum()
        previous_bs[self.franchises + 'w'] = previous_bs.groupby(['franchise'])[self.franchises + 'w'].cumsum()
        previous_bs[self.franchises + 'g'] = previous_bs[self.franchises + 'g'].replace(0, np.nan)

        previous_bs[self.bef_columns] = \
            previous_bs.groupby(['season', 'team_id'], as_index=False)[
                self.bef_columns].cumsum().reset_index(drop=True)

        previous_bs[self.bef_columns] = previous_bs[self.bef_columns].fillna(0)

        previous_bs['rest'] = previous_bs['date'].diff().apply(lambda x: x.days)
        previous_bs.loc[abs(previous_bs['rest']) > 50, 'rest'] = np.NaN

        previous_bs['streak'] = previous_bs.groupby(['season', 'team_id'], as_index=
        False)['victory'].rolling(min_periods=streak, window=streak).sum().reset_index(drop=True)

        previous_bs['games'] = previous_bs.groupby(['season', 'team_id'], as_index=False).cumcount() + 1
        previous_bs['cum_victories'] = previous_bs.groupby(['season', 'team_id'], as_index=False)[
            'victory'].cumsum()
        previous_bs['win_pct_total'] = previous_bs['cum_victories'] / previous_bs['games']

        previous_bs['h_games'] = previous_bs.groupby(['season', 'team_id'], as_index=False)['home'].cumsum()
        previous_bs['h_cum_victories'] = previous_bs.apply(
            lambda row: 1 if row.victory == 1 and row.home == 1 else 0, axis=1)
        previous_bs['h_cum_victories'] = previous_bs.groupby(['season', 'team_id'], as_index=False)[
            'h_cum_victories'].cumsum()
        previous_bs['win_pct_home'] = previous_bs['h_cum_victories'] / previous_bs['h_games']

        previous_bs['a_games'] = previous_bs.apply(lambda row: 1 if row.home == 0 else 0, axis=1)
        previous_bs['a_games'] = previous_bs.groupby(['season', 'team_id'], as_index=False)['a_games'].cumsum()
        previous_bs['a_cum_victories'] = previous_bs.apply(
            lambda row: 1 if row.victory == 1 and row.home == 0 else 0, axis=1)
        previous_bs['a_cum_victories'] = previous_bs.groupby(['season', 'team_id'], as_index=False)[
            'a_cum_victories'].cumsum()
        previous_bs['win_pct_away'] = previous_bs['a_cum_victories'] / previous_bs['a_games']

        previous_bs['ort'] = 100 * previous_bs['pts'] / previous_bs['poss']
        previous_bs['drt'] = 100 * previous_bs['pt_allowed'] / previous_bs['poss']

        previous_bs['pace'] = 48 * previous_bs['poss'] * 2 / (2 * previous_bs['mp'] / 5)

        previous_bs['eFG'] = (previous_bs['fg'] + 0.5 * previous_bs['fg3']) / previous_bs['fga']
        previous_bs['ftR'] = previous_bs['ft'] / previous_bs['fga']
        previous_bs['tovR'] = previous_bs['tov'] / (
                previous_bs['fga'] + (previous_bs['fta'] * 0.44) + previous_bs['tov'])
        previous_bs['orbR'] = previous_bs['orb'] / (previous_bs['orb'] + previous_bs['drb_opp'])
        previous_bs['drbR'] = previous_bs['drb'] / (previous_bs['drb'] + previous_bs['orb_opp'])

        current = previous_bs.groupby('team_id', as_index=False).tail(1).reset_index(drop=True)
        current_season = np.unique(current.season)[-1]
        current = current[current.season == current_season]

        previous_bs[
            self.bef_columns + ['eFG', 'ftR', 'tovR', 'orbR', 'drbR', 'win_pct_total', 'win_pct_home',
                                'win_pct_away', 'pt_diff', 'pt_allowed', 'ort', 'drt', 'poss', 'pace']] = \
            previous_bs.groupby(['season', 'team_id'], as_index=False)[
                self.bef_columns + ['eFG', 'ftR', 'tovR', 'orbR', 'drbR', 'win_pct_total', 'win_pct_home',
                                    'win_pct_away', 'pt_diff', 'pt_allowed', 'ort', 'drt', 'poss', 'pace']].shift(1,
                                                                                                                  fill_value=0)
        previous_bs['streak'] = previous_bs.groupby(['season', 'team_id'])['streak'].shift(1)

        previous_bs['pytha'] = previous_bs['pts'] ** 16.5 / (
                previous_bs['pts'] ** 16.5 + previous_bs['pt_allowed'] ** 16.5)

        previous_bs['pt_diff'] = previous_bs['pt_diff'] / (previous_bs['games'] - 1)
        previous_bs['pt_allowed'] = previous_bs['pt_allowed'] / (previous_bs['games'] - 1)
        previous_bs['poss'] = previous_bs['poss'] / (previous_bs['games'] - 1)

        previous_bs['proj_win'] = (previous_bs['pt_diff'] * 2.7 + 41) / 82

        current['pytha']=current['pts'] ** 16.5 / (
                current['pts'] ** 16.5 + current['pt_allowed'] ** 16.5)

        current['pt_diff'] = current['pt_diff'] / (current['games'] - 1)
        current['pt_allowed'] = current['pt_allowed'] / (current['games'] - 1)
        current['poss'] = current['poss'] / (current['games'] - 1)

        current['proj_win'] = (current['pt_diff'] * 2.7 + 41) / 82

        previous_bs[['pt_diff', 'pt_allowed', 'poss', 'pytha', 'proj_win']] = previous_bs[
            ['pt_diff', 'pt_allowed', 'poss', 'pytha', 'proj_win']].fillna(0)

        previous_bs[teams] = previous_bs.groupby(['team_id'])[teams].shift(1)
        previous_bs[teams] = previous_bs[teams].astype(np.float64)
        previous_bs['head_to_head'] = previous_bs.apply(
            lambda row: row[str(row.fr_opp) + 'w'] / (row[str(row.fr_opp) + 'g']) if not pd.isnull(
                row.fr_opp) else row.fr_opp,
            axis=1)
        previous_bs['head_to_head'] = previous_bs['head_to_head'].fillna(0.5)
        previous_bs = previous_bs.drop(teams + self.columns +
                                       ['games', 'cum_victories', 'h_games', 'h_cum_victories', 'a_games',
                                        'a_cum_victories', 'drb_opp', 'orb_opp', 'opp', 'franchise', 'fr_opp', 'mp'],
                                       axis=1)
        current = current.drop(self.columns+['games', 'cum_victories', 'h_games', 'h_cum_victories', 'a_games',
                                'a_cum_victories', 'drb_opp', 'orb_opp', 'opp', 'franchise', 'fr_opp', 'mp'], axis=1)
        current['game_id'] = np.nan
        current['rest'] = np.nan
        current['home'] = np.nan
        current['victory'] = np.nan
        current['head_to_head'] = np.nan
        current.to_csv(self.path + 'current.csv', index=False)
        previous_bs.to_csv(self.path + 'before_game.csv', index=False)

    # Crea los conjuntos con los que trabajan los modelos a partir de las caracteristicas avanzadas calculadas
    # en la función anterior
    # Params:
    #   previous: dataframe con las estadisticas avanzadas
    def create_games(self, previous):
        h_team = previous.loc[previous['home'] == True].reset_index(drop=True)
        a_team = previous.loc[previous['home'] == False].reset_index(drop=True)
        games = self.join_ha(h_team, a_team)
        games.to_csv(self.path + 'games.csv', index=False)

    # Une las estadísticas conseguidas por cada equipo local y visitante para realizar una observación
    # para cada partido
    # Params:
    #   h_team: dataframe con las estadisticas de los equipos locales
    #   a_team: dataframe con las estadisticas de los equipos visitantes
    def join_ha(self, h_team, a_team):
        h_names = "h_" + h_team.columns
        h_team = h_team.rename(columns=dict(zip(h_team.columns, h_names)))

        a_names = "a_" + a_team.columns
        a_team = a_team.rename(columns=dict(zip(a_team.columns, a_names)))

        h_team = h_team.rename(
            columns={'h_game_id': 'game_id', 'h_season': 'season', 'h_date': 'date', 'h_victory': 'victory',
                     'h_head_to_head': 'head_to_head'})
        a_team = a_team.rename(columns={'a_game_id': 'game_id', 'a_season': 'season', 'a_date': 'date'})
        h_team = h_team.drop(['h_home', 'h_win_pct_away'], axis=1)
        a_team = a_team.drop(['a_home', 'a_victory', 'a_head_to_head', 'a_win_pct_home'], axis=1)
        ha_bs = pd.merge(h_team, a_team, on=['game_id', 'season', 'date'])

        ha_bs['expected'] = ha_bs['h_proj_win'] * (1 - ha_bs['a_proj_win']) / (
                ha_bs['h_proj_win'] * (1 - ha_bs['a_proj_win']) +
                ((1 - ha_bs['h_proj_win']) * ha_bs['a_proj_win']))
        ha_bs['expected'] = ha_bs['expected'].fillna(0)
        return ha_bs

    # Realiza operaciones para la actualizacion de los conjuntos de datos
    # Params:
    #   new_games: dataframe con partidos que no han sido tratados
    def add_games(self, new_games):
        if not new_games.empty:
            aux = self.create_traditional_boxscore(new_games, ret=True)
            aux2 = self.create_team_totals(aux, ret=True)
            cumulative = self.get_team_totals()
            new = cumulative.append(aux2, ignore_index=True)

            start_time = timeit.default_timer()
            self.create_before(new)
            elapsed = timeit.default_timer() - start_time
            print('before_game: ' + str(elapsed))

            before_bs = self.get_before()
            start_time = timeit.default_timer()
            self.create_games(before_bs)
            elapsed = timeit.default_timer() - start_time
            print('games: ' + str(elapsed))

            print("Archivo  actualizado")

    # Compara los partidos del conjunto de datos original y el de games, y actualiza la base de datos
    # que contiene las estadisticas avanzadas de cada partido, es decir, el conjunto de games
    def refresh(self):
        ad = pd.read_csv(self.path + 'games.csv')
        ad['date'] = pd.to_datetime(ad['date'])
        last_date = ad.sort_values('date')['date'].tail(1).values[0]
        bs = self.get_raw()
        games = bs[bs.date > last_date]
        self.add_games(games)

    # Realiza todo el procesamiento desde el conjunto de datos original hasta el conjunto de games
    def main(self):
        db=self.get_raw()
        start_time = timeit.default_timer()
        self.create_traditional_boxscore(db)
        elapsed = timeit.default_timer() - start_time
        print('boxscore: ' + str(elapsed))

        trad_bs = self.get_boxscore()
        start_time = timeit.default_timer()
        self.create_team_totals(trad_bs)
        elapsed = timeit.default_timer() - start_time
        print('team_totals: ' + str(elapsed))

        team_bs = self.get_team_totals()
        start_time = timeit.default_timer()
        self.create_before(team_bs)
        elapsed = timeit.default_timer() - start_time
        print('before_game: ' + str(elapsed))

        before_bs = self.get_before()
        start_time = timeit.default_timer()
        self.create_games(before_bs)
        elapsed = timeit.default_timer() - start_time
        print('games: ' + str(elapsed))

    # Lee del almacenamiento local el conjunto de datos original
    def get_raw(self):
        aux = pd.read_csv(self.path+"raw_2000.csv")
        aux['date']=pd.to_datetime(aux['date'])
        return aux

    # Lee del almacenamiento local el conjunto de datos con los box score
    def get_boxscore(self):
        aux = pd.read_csv(self.path + 'boxscore.csv')
        aux['date'] = pd.to_datetime(aux['date'])
        return aux

    # Lee del almacenamiento local el conjunto de datos con las estadisticas por equipo
    def get_team_totals(self):
        aux = pd.read_csv(self.path + 'team_totals.csv')
        aux['date'] = pd.to_datetime(aux['date'])
        return aux

    # Lee del almacenamiento local la tabla con todos los nombres de franquicias
    def get_franchises(self):
        return pd.read_csv(self.path + 'franchises.csv')

    # Lee del almacenamiento local el conjunto de datos con los datos previos a cada partido
    def get_before(self):
        aux = pd.read_csv(self.path + 'before_game.csv')
        aux['date'] = pd.to_datetime(aux['date'])
        return aux

    # Lee del almacenamiento local el conjunto de datos con los datos de los partidos
    def get_games(self):
        aux = pd.read_csv(self.path + 'games.csv')
        aux['date'] = pd.to_datetime(aux['date'])
        return aux

    # Lee del almacenamiento local el conjunto de datos con los datos actuales de cada equipo
    def get_current(self):
        aux = pd.read_csv(self.path + 'current.csv')
        aux['date'] = pd.to_datetime(aux['date'])
        return aux


# Ejecucion de todo el procesamiento desde el archivo Raw hasta los datos de los partidos utilizados en
# los modelos predicitvos (puede tardar unos 12 minutos)
# GestorBBDD(path='../datos/').main()