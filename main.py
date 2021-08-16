# Imports
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import display
pd.options.mode.chained_assignment = None
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class Main:

    def __init__(self):

        path = "C:/Users/theerik/PycharmProjects/predictor/models/5585.txt"

        self.model = xgb.XGBClassifier()
        self.model.load_model(path)

        self.future = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/template.csv")


    def main(self):

        teams = list(self.future.keys())[:62]

        teams_set = set()
        i = 0
        while True:
            data = self.future.iloc[[i]]
            home = None
            away = None
            for team in teams:
                val = data[team].iloc[0]
                val = val.T
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
                break
            teams_set.add(home)
            if away in teams_set:
                print("away in set")
                break
            teams_set.add(away)
            pred = self.predict(data)

            # write to file
            row = f"{home},{away},{pred},-\n"
            with open("C:/Users/theerik/PycharmProjects/predictor/predictions/predictions.csv", 'a') as fd:
                fd.write(row)

            # show
            print(f"{i} Home: {home.ljust(14, ' ')} | Away: {away.ljust(14, ' ')} | {pred}  ")
            i += 1

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
    m.main()