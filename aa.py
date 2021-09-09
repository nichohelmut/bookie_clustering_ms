import google.auth
import pandas as pd
import pandas_gbq

from clustering import ArchetypalAnalysis


def read_bigquery(table_name):
    credentials, project_id = google.auth.default()
    df = pandas_gbq.read_gbq('select * from `footy-323809.statistics.{}`'.format(table_name),
                             project_id=project_id,
                             credentials=credentials,
                             location='europe-west1')

    return df


def write(df, project_id, output_dataset_id, output_table_name, credentials):
    print("write to bigquery")
    df.to_gbq(
        "{}.{}".format(output_dataset_id, output_table_name),
        project_id=project_id,
        if_exists="replace",
        credentials=credentials,
        progress_bar=None
    )
    print("Query complete. The table is updated.")



class AA:
    def __init__(self):
        pass

    def european_leagues(self):
        df_all = read_bigquery('international_teams_current')
        df_european_leagues = df_all.copy()

        return df_european_leagues

    # def climbers(self):
    #
    #     df_germany = bq_reader.read_bigquery('germany_teams')
    #
    #     df_germany_second = df_germany[df_germany['divison'] == 'second']
    #     df_fuerth = df_germany_second[df_germany_second['common_name'] == 'Greuther Fürth']
    #
    #     path_bl = os.path.join(path_to_t_stats, "germany-bundesliga-teams-2019-to-2020-stats.csv")
    #     df_temas_last = pd.read_csv(path_bl)
    #
    #     df_dusseldorf = df_temas_last[df_temas_last['common_name'] == 'Fortuna Düsseldorf']
    #     df_paderborn = df_temas_last[df_temas_last['common_name'] == 'Paderborn']
    #
    #     path_teams_last_first = os.path.join(path_to_t_stats, 'germany-bundesliga-teams-2018-to-2019-stats.csv')
    #     df_teams_last_first = pd.read_csv(path_teams_last_first)
    #
    #     df_nueremberg = df_teams_last_first[df_teams_last_first['common_name'] == 'Nürnberg']
    #     df_hannover = df_teams_last_first[df_teams_last_first['common_name'] == 'Hannover 96']
    #     df_stuttgart = df_teams_last_first[df_teams_last_first['common_name'] == 'Stuttgart']
    #
    #     path_teams_last_17_18 = os.path.join(path_to_t_stats, 'germany-bundesliga-teams-2017-to-2018-stats.csv')
    #     df_teams_last_17_18 = pd.read_csv(path_teams_last_17_18)
    #     df_hsv = df_teams_last_17_18[df_teams_last_17_18['common_name'] == 'Hamburger SV']
    #
    #     path_teams_last_16_17 = os.path.join(path_to_t_stats, 'germany-bundesliga-teams-2016-to-2017-stats.csv')
    #     df_teams_last_16_17 = pd.read_csv(path_teams_last_16_17)
    #     df_darmstadt = df_teams_last_16_17[df_teams_last_16_17['common_name'] == 'Darmstadt 98']
    #     df_ingolstadt = df_teams_last_16_17[df_teams_last_16_17['common_name'] == 'Ingolstadt']
    #
    #     df_league_climbers = pd.concat(
    #         [df_nueremberg, df_bielefeld, df_hannover, df_stuttgart, df_hsv, df_darmstadt, df_ingolstadt, df_dusseldorf,
    #          df_paderborn], sort=False)
    #     df_league_climbers.reset_index(inplace=True)
    #     df_league_climbers.drop("index", axis=1, inplace=True)
    #     df_all_climbers = df_league_climbers.copy()
    #
    #     return df_all_climbers

    # def top_leagues_with_climbers(self):
    #     # to make sure both input dfs have the same columns
    #     df_all = pd.concat([self.european_leagues()[self.climbers().columns], self.climbers()], sort=False)
    #     df_all.reset_index(inplace=True)
    #     df_all.drop("index", axis=1, inplace=True)
    #     df_total = df_all.copy()
    #
    #     return df_total

    def matrix(self):
        df_all = self.european_leagues()
        df_all.set_index("team_name", inplace=True)
        df_all = df_all.T

        df_teams_numerical = df_all.iloc[8:, :]

        df_norm = (df_teams_numerical - df_teams_numerical.min()) / (
                df_teams_numerical.max() - df_teams_numerical.min())

        X = df_norm.to_numpy()

        return X

    def aa_analysis(self):
        # TODO: FIX 5 ARCHETYPES LIMIT
        archetypal = ArchetypalAnalysis(n_archetypes=5, iterations=10, tmax=300)
        model = archetypal.fit(self.matrix())

        return model

    def archetypal_transform(self):
        A = self.aa_analysis().transform(self.matrix())

        return A

    def data_labels(self, A):
        teamsList = self.european_leagues()['team_name']
        temasColumnOrdering = {x: y for y, x in enumerate(teamsList)}
        d_labels = {v: k for k, v in temasColumnOrdering.items()}

        # fixes enumerate bug
        df_labels = pd.DataFrame.from_dict(d_labels, orient='index').reset_index(drop=True)
        labels = df_labels.to_dict()

        for i in range(0, len(labels[0])):
            print("{:40}".format(labels[0][i]), end='')
            for j in A[:, i]:
                print("{:.3f} ".format(j), end='')
            print("")

    def run(self):
        A = self.archetypal_transform()
        df_aa_result = pd.DataFrame(data=A.T)

        df_aa_result.columns = list(df_aa_result.columns.map(str))
        string = 'aa_'
        df_aa_result.columns = [string + x for x in df_aa_result.columns]

        df_teams_with_aa = pd.concat([self.european_leagues(), df_aa_result], axis=1)
        df_teams_only_aa = df_teams_with_aa.iloc[:, 280:]
        df_teams_only_aa['common_name'] = df_teams_with_aa['common_name']
        self.data_labels(A)
        credentials, project_id = google.auth.default()

        write(df_teams_only_aa, project_id, 'statistics', 'aa_results', credentials)
