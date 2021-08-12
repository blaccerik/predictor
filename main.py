# Imports
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import display

class Main:

    def __init__(self):

        path = "C:/Users/theerik/PycharmProjects/predictor/models/5447.txt"

        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        self.future = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/editedfuture.csv")
    #     self.check()
    #
    # def check(self):
    #     places = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/data/allplaces/data.csv")
    #     futuregames = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/editedfuture.csv")
    #     all_teams = futuregames.HomeTeam.unique()
    #
    #     self.future = futuregames
    #
    #     self.teams = []
    #     for team in all_teams:
    #         if team in places.Team.unique():
    #             self.teams.append(team)
    #         else:
    #             # no data
    #             if team == "Brentford":
    #                 continue
    #             print(team)
    #             raise Exception


    def main(self, week):
        start = (week - 1) * 10
        end = week * 10

        # change number if there are more teams
        teams = list(self.future.keys())[:62]

        teams_set = set()

        for i in range(start, end):
            data = self.future.iloc[[i]]
            home = None
            away = None
            for team in teams:
                val = data[team].iloc[0]
                if val == 1:
                    if "Home" in team:
                        home = team[9:]
                    elif "Away" in team:
                        away = team[9:]
            if home is None or away is None:
                print("One team is None")
                raise Exception
            if home in teams_set:
                print("home in set")
                raise Exception
            teams_set.add(home)
            if away in teams_set:
                print("home in set")
                raise Exception
            teams_set.add(away)
            pred = self.predict(data)
            # if error then team not found
            print(f"{i} Home: {home.ljust(14, ' ')} | Away: {away.ljust(14, ' ')} | {pred}  ")

    def predict(self, data):
        pred = self.model.predict(data)
        if pred == 0:
            return "H"
        elif pred == 1:
            return "A"
        else:
            return "D"

if __name__ == '__main__':
    m = Main()
    m.main(week=1)