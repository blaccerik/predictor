# Imports
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import display

class Main:

    def __init__(self):

        path = "C:/Users/theerik/PycharmProjects/predictor/models/6046.txt"

        self.model = xgb.XGBClassifier()
        self.model.load_model(path)

        future = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/future.csv")
        future = future[future['Check'] == False]
        self.future = future.drop(["Check"], axis=1)

        self.template = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/template.csv")


    def main(self):
        print(self.future)

        teams_set = set()
        i = 0
        while True:
            data = self.future.iloc[[i]]
            home = data["HomeTeam"]
            away = data["AwayTeam"]
            print(home, away)

            # todo update template and get prediction

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
            i += 1
            break

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