# Imports
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import display

class Main:

    def __init__(self):

        path = "model.txt"

        self.model = xgb.XGBClassifier()
        self.model.load_model(path)

        # todo download list of teams

    def main(self, hometeam, awayteam):
        # todo create list of teams
        pass

if __name__ == '__main__':
    m = Main()
    m.main()