import numpy as np
import pandas as pd
import requests
try:
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)

def read(s, link):
    html = s.get(link).text
    parsed_html = BeautifulSoup(html, features="html.parser")
    text = parsed_html.find_all("table", {"class": "table table-hover table-condensed"})
    team = []
    rank = []
    n = 0
    for i in text[0].find("tbody").find_all("tr"):
        n += 1
        a = i.find("a")
        name = a.__dict__['contents'][0]
        # todo remove FC from name
        team.append(name)
        rank.append(n)
    return team, rank

def main():
    s = requests.Session()
    all_data = []
    years = [2017, 2018, 2019, 2020, 2021]
    for i in range(5):
        n = 16 + i
        link = f"https://footballdatabase.com/league-scores-tables/england-premier-league-20{n}-{n + 1}"
        team, rank = read(s, link)
        more_data = pd.DataFrame()
        more_data["Team"] = team
        more_data["Rank"] = rank
        all_data.append(more_data)

    path = "/data/"

    # all_data = [data1, data2, data3]
    team_set = set()
    # get all the teams
    for data in all_data:
        for team in data.Team:
            team_set.add(team)

    # sort
    team_set = list(team_set)
    team_set.sort()

    # create "2d" array
    final = pd.DataFrame()
    final["Team"] = team_set
    for i in range(len(all_data)):
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
