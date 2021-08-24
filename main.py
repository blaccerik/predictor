# Imports
import pandas as pd
import xgboost as xgb
from datetime import date
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import display
from filter import Filter
from trainer import Trainer
from results import main as results_main
# from results import
pd.options.mode.chained_assignment = None
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class Main:
    def __init__(self):

        path = "C:/Users/theerik/PycharmProjects/predictor/models/5770.txt"

        self.model = xgb.XGBClassifier()
        self.model.load_model(path)

        self.write = self.check()

        self.future = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/template.csv")

    def check(self):
        # update results
        results_main()
        data = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/data/results/results.csv")
        first_date = data["Date"].iloc[-1]
        today = date.today().strftime("%d/%m/%Y")

        date_format = "%d/%m/%Y"
        a = datetime.strptime(first_date, date_format)
        b = datetime.strptime(today, date_format)
        delta = b - a
        if delta.days > 7:
            print("Results are too old")
            # go to results.py file and then click on the link
            # update web_source/results.txt and run results.py
            # this will generate a new results.csv file with up to date results

            # also it is good that you retrain your model to keep it fresh
            raise Exception

        results_size = data.shape[0]
        if results_size % 10 != 0:
            print("Not all this weeks games are played")
            # just wait till games are over

        # update predictions "real" results
        data2 = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/predictions/predictions.csv")
        pred_size = data2.shape[0]
        if "-" in data2["Real"].unique():
            for index, row in data.iterrows():
                home = row["HomeTeam"]
                away = row["AwayTeam"]
                ftr = row["FTR"]
                data2.loc[((data2["Home"] == home) & (data2["Away"] == away)), "Real"] = ftr
            data2.to_csv("C:/Users/theerik/PycharmProjects/predictor/predictions/predictions.csv", index=False)
        # update template
        f = Filter()
        f.future()
        t = Trainer()
        return results_size == pred_size

    def main(self):
        self.predict()

    def predict(self):

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
            pred, prob = self.translate_predict(data)

            # write to file
            home_win = "-"
            home_win_prob = "-"
            if self.write:
                row = f"{home},{away},{pred},-,{prob},{home_win}, {home_win_prob}\n"
                with open("C:/Users/theerik/PycharmProjects/predictor/predictions/predictions.csv", 'a') as fd:
                    fd.write(row)

            # show
            print(f"{i} Home: {home.ljust(14, ' ')} | Away: {away.ljust(14, ' ')} | {pred} | {prob}%  ")
            i += 1

    def translate_predict(self, data):
        pred = int(self.model.predict(data))
        prob = int(max(self.model.predict_proba(data)[0]) * 100)
        return Filter.number_to_string[pred], prob

if __name__ == '__main__':
    m = Main()
    m.main()