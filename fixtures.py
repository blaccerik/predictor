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

    # https://www.premierleague.com/fixtures

    with open("web_source/fictures.txt") as f:
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
                if home in diff_names:
                    home = diff_names[home]
                if away in diff_names:
                    away = diff_names[away]
                home_teams.append(home)
                away_teams.append(away)
                dates.append(date)
                ftrs.append("D")
                fthgs.append(0)
                ftags.append(0)
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
    final.to_csv("data/futureGames/fixtures.csv", index=False)


if __name__ == '__main__':
    main()

