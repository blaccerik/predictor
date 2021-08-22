import numpy as np
import pandas as pd
from datetime import datetime as dt
import itertools
from IPython.display import display
pd.options.mode.chained_assignment = None  # default='warn'
desired_width=320
pd.set_option('display.width', desired_width)
# np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import fixtures as fix

class Filter:

    path = "C:/Users/theerik/PycharmProjects/predictor/data/"
    weeks = 38
    teams_nr = 20
    games_nr = 380
    games_in_week = 10

    string_to_number = {
        "H": 0,
        "D": 1,
        "A": 2,
        "W": 0,
        "L": 2,
        "N": 3
    }

    number_to_string = {
        0: "H",
        1: "D",
        2: "A"
    }

    def parse_date(self, date):
        if date == '':
            return None
        else:
            try:
                return dt.strptime(date, '%d/%m/%Y').date()
            except ValueError:
                return dt.strptime(date, '%d/%m/%y').date()

    def ftr_number(self, string):
        return self.string_to_number[string]

    def home_win(self, string):
        if string == 'H':
            return 1
        else:
            return 0

    def away_win(self, string):
        if string == 'A':
            return 1
        else:
            return 0

    def get_goals(self, playing_stat):
        """
        Get goals for each week
        FTHG - Full Time Home Team Goals
        FTAG - Full Time Away Team Goals
        """

        # Create a dictionary with team names as keys
        teams_scored = {}
        teams_conceded = {}
        for i in playing_stat.groupby('HomeTeam').mean().transpose().columns:
            teams_scored[i] = []
            teams_conceded[i] = []
        for i in playing_stat.groupby('AwayTeam').mean().transpose().columns:
            teams_scored[i] = []
            teams_conceded[i] = []

        # the value corresponding to keys is a list containing the match location.
        for i in range(len(playing_stat)):
            datarow = playing_stat.iloc[i]
            home_team_goals_scored = datarow['FTHG']
            away_team_goals_scored = datarow['FTAG']
            teams_scored[datarow.HomeTeam].append(home_team_goals_scored)
            teams_scored[datarow.AwayTeam].append(away_team_goals_scored)
            teams_conceded[datarow.HomeTeam].append(away_team_goals_scored)
            teams_conceded[datarow.AwayTeam].append(home_team_goals_scored)

        # all data should be self.weeks long
        for i in teams_scored:
            val1 = teams_scored[i]
            if len(val1) < self.weeks:
                val1.append(0)
            val2 = teams_conceded[i]
            if len(val2) < self.weeks:
                val2.append(0)

        # Create a dataframe for goals scored
        # rows: teams
        # columns: week
        size = self.weeks + 1
        GoalsScored = pd.DataFrame(data=teams_scored, index=[i for i in range(1, size)]).transpose()
        GoalsConceded = pd.DataFrame(data=teams_conceded, index=[i for i in range(1, size)]).transpose()

        # prevent error
        GoalsScored[0] = 0
        GoalsConceded[0] = 0

        # add goals together
        for i in range(2, size):
            GoalsScored[i] = GoalsScored[i] + GoalsScored[i - 1]
            GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i - 1]

        return GoalsScored, GoalsConceded

    def get_points(self, result):
        if result == 'W':
            return 3
        elif result == 'D':
            return 1
        else:
            return 0

    def get_cuml_points(self, matches):

        matches_points = matches.applymap(self.get_points)

        for i in range(2, self.weeks + 1):
            matches_points[i] = matches_points[i] + matches_points[i - 1]

        matches_points.insert(column=0, loc=0, value=[0 * i for i in range(self.teams_nr)])
        return matches_points

    def get_goals_s_c(self, playing_stat):
        """
        find the following values:
        HTGS, ATGS, HTGC, ATGC
        """
        GoalsScored, GoalsConceded = self.get_goals(playing_stat)

        # print(GoalsScored)

        j = 0
        home_team_goals_scored = []
        away_team_goals_scored = []
        home_team_goals_conceded = []
        away_team_goals_conceded = []
        for i in range(self.games_nr):
            datarow = playing_stat.iloc[i]
            ht = datarow.HomeTeam
            at = datarow.AwayTeam
            home_team_goals_scored.append(GoalsScored.loc[ht][j])
            away_team_goals_scored.append(GoalsScored.loc[at][j])
            home_team_goals_conceded.append(GoalsConceded.loc[ht][j])
            away_team_goals_conceded.append(GoalsConceded.loc[at][j])
            if ((i + 1) % self.games_in_week) == 0:
                j = j + 1
        playing_stat['HTGS'] = home_team_goals_scored
        playing_stat['ATGS'] = away_team_goals_scored
        playing_stat['HTGC'] = home_team_goals_conceded
        playing_stat['ATGC'] = away_team_goals_conceded

        return playing_stat

    def get_matches(self, playing_stat):
        # Create a dictionary with team names as keys
        teams = {}
        for i in playing_stat.groupby('HomeTeam').mean().T.columns:
            teams[i] = []
        for i in playing_stat.groupby('AwayTeam').mean().T.columns:
            teams[i] = []

        # the value corresponding to keys is a list containing the match result
        for i in range(len(playing_stat)):
            if playing_stat.iloc[i].FTR == self.string_to_number['H']:
                teams[playing_stat.iloc[i].HomeTeam].append('W')
                teams[playing_stat.iloc[i].AwayTeam].append('L')
            elif playing_stat.iloc[i].FTR == self.string_to_number['A']:
                teams[playing_stat.iloc[i].AwayTeam].append('W')
                teams[playing_stat.iloc[i].HomeTeam].append('L')
            elif playing_stat.iloc[i].FTR == self.string_to_number['D']:
                teams[playing_stat.iloc[i].AwayTeam].append('D')
                teams[playing_stat.iloc[i].HomeTeam].append('D')
            else:
                raise Exception

        return pd.DataFrame(data=teams, index=[i for i in range(1, self.weeks + 1)]).T

    def get_all_points(self, playing_stat):
        matches = self.get_matches(playing_stat)
        cum_pts = self.get_cuml_points(matches)
        HTP = []
        ATP = []
        j = 0
        for i in range(self.games_nr):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam
            HTP.append(cum_pts.loc[ht][j])
            ATP.append(cum_pts.loc[at][j])

            if ((i + 1) % self.games_in_week) == 0:
                j = j + 1

        playing_stat['HTP'] = HTP
        playing_stat['ATP'] = ATP
        return playing_stat

    def get_form(self, playing_stat, num):
        form = self.get_matches(playing_stat)
        form_final = form.copy()
        for i in range(num, self.weeks + 1):
            form_final[i] = ''
            j = 0
            while j < num:
                form_final[i] += form[i - j]
                j += 1
        return form_final

    def add_form(self, playing_stat, num):
        form = self.get_form(playing_stat, num)
        # dont know wins/losses at the beginning
        h = ['N' for i in range(num * self.games_in_week)]
        a = ['N' for i in range(num * self.games_in_week)]

        j = num
        for i in range((num * self.games_in_week), self.games_nr):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam

            past = form.loc[ht][j]  # get past n results
            h.append(past[num - 1])  # 0 index is most recent

            past = form.loc[at][j]
            a.append(past[num - 1])

            if ((i + 1) % 10) == 0:
                j = j + 1

        # if you want to calculate lets say 2 weeks worth of AHD strings but there is only 1 week of data
        # then this prevents it
        while len(h) > self.games_nr:
            del h[-1]
        while len(a) > self.games_nr:
            del a[-1]

        playing_stat['HM' + str(num)] = h
        playing_stat['AM' + str(num)] = a

        return playing_stat

    def add_prev_matches(self, playing_statistics):
        playing_statistics = self.add_form(playing_statistics, 1)
        playing_statistics = self.add_form(playing_statistics, 2)
        playing_statistics = self.add_form(playing_statistics, 3)
        playing_statistics = self.add_form(playing_statistics, 4)
        playing_statistics = self.add_form(playing_statistics, 5)
        return playing_statistics

    def get_last(self, playing_stat, standings, year):
        HomeTeamLP = []
        AwayTeamLP = []
        for i in range(self.games_nr):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam

            # if no data then check spelling
            # if still no data then add it to allplaces/data.csv
            HomeTeamLP.append(standings.loc[ht][year])
            AwayTeamLP.append(standings.loc[at][year])

        playing_stat['HTLP'] = HomeTeamLP
        playing_stat['ATLP'] = AwayTeamLP
        return playing_stat

    def get_mw(self, playing_stat):
        j = 1
        MatchWeek = []
        for i in range(self.games_nr):
            MatchWeek.append(j)
            if ((i + 1) % self.games_in_week) == 0:
                j = j + 1
        playing_stat['MW'] = MatchWeek
        return playing_stat

    def get_form_points(self, string):
        sum = 0
        for letter in string:
            sum += self.get_points(letter)
        return sum

    def streaks(self, games_stats):
        # find win/lose streaks
        self.nr, self.part = 3, "WWW"
        games_stats['HTWS3'] = games_stats['HTFPS'].apply(self.detect)
        self.nr, self.part = 5, "WWWWW"
        games_stats['HTWS5'] = games_stats['HTFPS'].apply(self.detect)
        self.nr, self.part = 3, "LLL"
        games_stats['HTLS3'] = games_stats['HTFPS'].apply(self.detect)
        self.nr, self.part = 5, "LLLLL"
        games_stats['HTLS5'] = games_stats['HTFPS'].apply(self.detect)
        self.nr, self.part = 3, "WWW"
        games_stats['ATWS3'] = games_stats['ATFPS'].apply(self.detect)
        self.nr, self.part = 5, "WWWWW"
        games_stats['ATWS5'] = games_stats['ATFPS'].apply(self.detect)
        self.nr, self.part = 3, "LLL"
        games_stats['ATLS3'] = games_stats['ATFPS'].apply(self.detect)
        self.nr, self.part = 5, "LLLLL"
        games_stats['ATLS5'] = games_stats['ATFPS'].apply(self.detect)

        return games_stats

    def detect(self, string):
        if string[-self.nr:] == self.part:
            return 1
        else:
            return 0

    def binary(self, string):
        if string == 'H':
            return 0
        elif string == "A":
            return 1
        else:
            return 2

    def modify_data(self, data, check=False):

        # keep only needed columns
        if not check:
            data["Check"] = True
        columns_req = ["Check", 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

        games_stats = data[columns_req]

        # adjust date
        games_stats.Date = games_stats.Date.apply(self.parse_date)
        # change ftr and its "relatives"
        games_stats["FTRH"] = games_stats.FTR.apply(self.home_win)
        games_stats["FTRA"] = games_stats.FTR.apply(self.away_win)
        games_stats.FTR = games_stats.FTR.apply(self.ftr_number)

        # get scored and conceled goals to that point
        games_stats = self.get_goals_s_c(games_stats)

        # get all points from prev games up to that point
        games_stats = self.get_all_points(games_stats)

        # get results for last 5 games
        games_stats = self.add_prev_matches(games_stats)

        # Rearranging columns
        # NOT NEEDED
        cols = ["Check", 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', "FTRH", "FTRA", 'HTGS', 'ATGS', 'HTGC',
                'ATGC', 'HTP', 'ATP', 'HM1', 'HM2', 'HM3', 'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5']
        games_stats = games_stats[cols]

        # read standings
        standings = pd.read_csv(self.path + "allplaces/data.csv")
        standings.set_index(['Team'], inplace=True)
        standings.replace(0, self.teams_nr, inplace=True)

        # add last year place
        games_stats = self.get_last(games_stats, standings, 0)

        # add match week
        games_stats = self.get_mw(games_stats)

        # get string of last 5 games
        games_stats['HTFPS'] = games_stats['HM1'] + games_stats['HM2'] + games_stats['HM3'] + games_stats['HM4'] + games_stats['HM5']
        games_stats['ATFPS'] = games_stats['AM1'] + games_stats['AM2'] + games_stats['AM3'] + games_stats['AM4'] + games_stats['AM5']

        # change hm and am to numbers
        string_list = ["HM", "AM"]
        for part in string_list:
            for i in range(1, 6):
                text = f"{part}{i}"
                games_stats[text] = games_stats[text].apply(self.ftr_number)

        # get points of last 5 games
        games_stats['HTFP'] = games_stats['HTFPS'].apply(self.get_form_points)
        games_stats['ATFP'] = games_stats['ATFPS'].apply(self.get_form_points)

        games_stats = self.streaks(games_stats)

        # Get Goal Difference
        games_stats['HTGD'] = games_stats['HTGS'] - games_stats['HTGC']
        games_stats['ATGD'] = games_stats['ATGS'] - games_stats['ATGC']

        # Diff in points
        games_stats['DP'] = games_stats['HTP'] - games_stats['ATP']
        games_stats['DFP'] = games_stats['HTFP'] - games_stats['ATFP']

        # Diff in last year positions
        games_stats['DLP'] = games_stats['HTLP'] - games_stats['ATLP']

        return games_stats

    def main(self):
        # https://www.football-data.co.uk/englandm.php

        # read past results
        data1 = pd.read_csv(self.path + "matches/2020-2021.csv")
        data2 = pd.read_csv(self.path + "matches/2019-2020.csv")
        data3 = pd.read_csv(self.path + "matches/2018-2019.csv")
        data4 = pd.read_csv(self.path + "matches/2017-2018.csv")
        data5 = pd.read_csv(self.path + "matches/2016-2017.csv")
        games_stats1 = self.modify_data(data1)
        games_stats2 = self.modify_data(data2)
        games_stats3 = self.modify_data(data3)
        games_stats4 = self.modify_data(data4)
        games_stats5 = self.modify_data(data5)

        # add all the things toghther
        playing_stat = pd.concat([
            games_stats1,
            games_stats2,
            games_stats3,
            games_stats4,
            games_stats5
        ], ignore_index=True)

        display(playing_stat.head(30))
        playing_stat.to_csv(self.path + "final/final.csv", index=False)


    def future(self):
        # read results
        results = pd.read_csv(self.path + "results/results.csv")
        future = pd.read_csv(self.path + "futureGames/fixtures.csv")

        # self.weeks = len(results) // 10 + 1
        # self.games_nr = len(results)
        # games_results = self.modify_data(results)
        # self.weeks = 38
        # self.games_nr = 380

        future["Check"] = False

        # change "future" results
        for index, row in results.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            home_score = row["FTHG"]
            away_score = row["FTAG"]
            res = row["FTR"]
            future.loc[((future["HomeTeam"] == home) & (future["AwayTeam"] == away)), ["FTHG", "FTAG", "FTR", "Check"]] = \
                home_score, away_score, res, True

        games_future = self.modify_data(future, check=True)
        games_future.to_csv(self.path + "futureGames/future.csv", index=False)
        print("Updated future.csv")

if __name__ == '__main__':
    f = Filter()
    f.future()
    # f.main()
