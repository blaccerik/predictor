import calendar
from datetime import datetime as dt
import numpy as np
import pandas as pd
import requests
try:
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

diff_names = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
}


def read():
    final = pd.DataFrame()
    with open("web_source/results.txt") as f:
        parsed_html = BeautifulSoup(f, features="html.parser")
        text = parsed_html.find_all("div", {"class": "fixtures__matches-list"})
        home_teams = []
        away_teams = []
        dates = []
        ftrs = []
        fthgs = []
        ftags = []

        months = {month: index for index, month in enumerate(calendar.month_name) if month}

        for row in text:
            datestring = row["data-competition-matches-list"]
            a = datestring[datestring.find("day") + 4:]
            lista = a.split(" ")
            day = lista[0]
            month = months[lista[1]]
            year = lista[2]
            date = f"{day}/{month}/{year}"
            row2 = row.find_all("li", {"class": "matchFixtureContainer"})
            for i in row2:
                home = i["data-home"]
                away = i["data-away"]
                score = i.find("span", {"class": "score"})
                score_l = score.text.split("-")
                home_score = int(score_l[0])
                away_score = int(score_l[1])
                if home in diff_names:
                    home = diff_names[home]
                if away in diff_names:
                    away = diff_names[away]
                home_teams.append(home)
                away_teams.append(away)
                dates.append(date)
                if home_score > away_score:
                    letter = "H"
                elif away_score > home_score:
                    letter = "A"
                else:
                    letter = "D"
                ftrs.append(letter)
                fthgs.append(home_score)
                ftags.append(away_score)
        # revers so that new games are in bottom
        dates.reverse()
        home_teams.reverse()
        away_teams.reverse()
        ftrs.reverse()
        fthgs.reverse()
        ftags.reverse()

        final["Date"] = dates
        final["HomeTeam"] = home_teams
        final["AwayTeam"] = away_teams
        final["FTHG"] = fthgs
        final["FTAG"] = ftags
        final["FTR"] = ftrs
    return final

def main():
    # https://www.premierleague.com/fixtures
    final = read()
    final.to_csv("data/results/results.csv", index=False)


if __name__ == '__main__':
    main()