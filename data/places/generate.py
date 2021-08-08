import numpy as np
import pandas as pd
from datetime import datetime as dt
import itertools
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def main():
    path = "C:/Users/theerik/PycharmProjects/predictor/data/"

    data3 = pd.read_csv(path + "places/2020-2021.csv")
    data2 = pd.read_csv(path + "places/2019-2020.csv")
    data1 = pd.read_csv(path + "places/2018-2019.csv")

    # years are counted when the season ended
    years = [2019, 2020, 2021]

    all_data = [data1, data2, data3]
    team_set = set()
    # get all the teams
    for data in all_data:
        for team in data.Team:
            team_set.add(team)
    # sort
    team_set = list(team_set)
    team_set.sort()

    # create 2d array
    final = pd.DataFrame()
    final["Team"] = team_set
    for i in range(len(years)):
        data = all_data[i]
        year = years[i]
        final[year] = np.zeros(len(team_set), np.int32)
        for j in range(0, 20):
            team = data.Team[j]
            rank = data.Rank[j]
            nr = final[final["Team"] == team].index.values
            final[year][nr] = rank
    final.to_csv(path + "/allplaces/data.csv", index=False)

if __name__ == '__main__':
    main()
