import numpy as np
import pandas as pd
from datetime import datetime as dt
import itertools
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class Filter:

    path = "C:/Users/theerik/PycharmProjects/predictor/data/"
    weeks = 38
    teams_nr = 20

    def parse_date(self, date):
        if date == '':
            return None
        else:
            return dt.strptime(date, '%d/%m/%Y').date()

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

        # the value corresponding to keys is a list containing the match location.
        for i in range(len(playing_stat)):
            datarow = playing_stat.iloc[i]
            home_team_goals_scored = datarow['FTHG']
            away_team_goals_scored = datarow['FTAG']
            teams_scored[datarow.HomeTeam].append(home_team_goals_scored)
            teams_scored[datarow.AwayTeam].append(away_team_goals_scored)
            teams_conceded[datarow.HomeTeam].append(away_team_goals_scored)
            teams_conceded[datarow.AwayTeam].append(home_team_goals_scored)

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

        j = 0
        home_team_goals_scored = []
        away_team_goals_scored = []
        home_team_goals_conceded = []
        away_team_goals_conceded = []
        for i in range(self.weeks * 10):
            datarow = playing_stat.iloc[i]
            ht = datarow.HomeTeam
            at = datarow.AwayTeam
            home_team_goals_scored.append(GoalsScored.loc[ht][j])
            away_team_goals_scored.append(GoalsScored.loc[at][j])
            home_team_goals_conceded.append(GoalsConceded.loc[ht][j])
            away_team_goals_conceded.append(GoalsConceded.loc[at][j])
            if ((i + 1) % 10) == 0:
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

        # the value corresponding to keys is a list containing the match result
        for i in range(len(playing_stat)):
            if playing_stat.iloc[i].FTR == 'H':
                teams[playing_stat.iloc[i].HomeTeam].append('W')
                teams[playing_stat.iloc[i].AwayTeam].append('L')
            elif playing_stat.iloc[i].FTR == 'A':
                teams[playing_stat.iloc[i].AwayTeam].append('W')
                teams[playing_stat.iloc[i].HomeTeam].append('L')
            else:
                teams[playing_stat.iloc[i].AwayTeam].append('D')
                teams[playing_stat.iloc[i].HomeTeam].append('D')

        return pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T

    def get_all_points(self, playing_stat):
        matches = self.get_matches(playing_stat)
        cum_pts = self.get_cuml_points(matches)
        HTP = []
        ATP = []
        j = 0
        for i in range(self.weeks * 10):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam
            HTP.append(cum_pts.loc[ht][j])
            ATP.append(cum_pts.loc[at][j])

            if ((i + 1) % 10) == 0:
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
        h = ['M' for i in range(num * 10)]  # since form is not available for n MW (n*10)
        a = ['M' for i in range(num * 10)]

        j = num
        for i in range((num * 10), self.weeks * 10):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam

            past = form.loc[ht][j]  # get past n results
            h.append(past[num - 1])  # 0 index is most recent

            past = form.loc[at][j]
            a.append(past[num - 1])

            if ((i + 1) % 10) == 0:
                j = j + 1

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

    def get_last(self, playing_stat, Standings, year):
        HomeTeamLP = []
        AwayTeamLP = []
        for i in range(380):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam

            # if no data on prev years about that team
            try:
                HomeTeamLP.append(Standings.loc[ht][year])
            except KeyError:
                HomeTeamLP.append(18)
            try:
                AwayTeamLP.append(Standings.loc[at][year])
            except KeyError:
                AwayTeamLP.append(18)
            # print(HomeTeamLP)
        playing_stat['HomeTeamLP'] = HomeTeamLP
        playing_stat['AwayTeamLP'] = AwayTeamLP
        return playing_stat

    def get_mw(self, playing_stat):
        j = 1
        MatchWeek = []
        for i in range(self.weeks * 10):
            MatchWeek.append(j)
            if ((i + 1) % 10) == 0:
                j = j + 1
        playing_stat['MW'] = MatchWeek
        return playing_stat

    def modify_data(self, data):

        # adjust date
        data.Date = data.Date.apply(self.parse_date)

        # keep only needed columns
        columns_req = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        games_stats = data[columns_req]

        # get scored and conceled goals
        games_stats = self.get_goals_s_c(games_stats)

        # get all points from prev games up to that point
        games_stats = self.get_all_points(games_stats)

        # get results for last 5 games
        games_stats = self.add_prev_matches(games_stats)

        # Rearranging columns
        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP',
                'HM1', 'HM2', 'HM3', 'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5']
        games_stats = games_stats[cols]

        # read standings
        Standings = pd.read_csv(self.path + "allplaces/data.csv")
        Standings.set_index(['Team'], inplace=True)
        Standings.replace(0, 18, inplace=True)

        # add last year place
        games_stats = self.get_last(games_stats, Standings, 0)

        # add match week
        games_stats = self.get_mw(games_stats)

        return games_stats

    def get_form_points(self, string):
        sum = 0
        for letter in string:
            sum += self.get_points(letter)
        return sum

    def detect(self, string):
        if string[-self.nr:] == self.part:
            return 1
        else:
            return 0

    def binary(self, string):
        if string == 'H':
            return 'H'
        else:
            return 'NH'

    def main(self):
        # data1 = pd.read_csv(path + "1516.csv")
        data1 = pd.read_csv(self.path + "matches/2020-2021.csv")
        # data2 = pd.read_csv(path + "1819.csv")
        # data3 = pd.read_csv(path + "1920.csv")
        # data4 = pd.read_csv(path + "2021.csv")

        games_stats1 = self.modify_data(data1)

        # add all the things toghther
        playing_stat = pd.concat([games_stats1, ], ignore_index=True)

        # get string of last 5 games
        playing_stat['HTFormPtsStr'] = playing_stat['HM1'] + playing_stat['HM2'] + playing_stat['HM3'] + playing_stat['HM4'] + playing_stat['HM5']
        playing_stat['ATFormPtsStr'] = playing_stat['AM1'] + playing_stat['AM2'] + playing_stat['AM3'] + playing_stat['AM4'] + playing_stat['AM5']

        # get points of last 5 games
        playing_stat['HTFormPts'] = playing_stat['HTFormPtsStr'].apply(self.get_form_points)
        playing_stat['ATFormPts'] = playing_stat['ATFormPtsStr'].apply(self.get_form_points)

        # find win/lose streaks
        self.nr, self.part = 3, "WWW"
        playing_stat['HTWinStreak3'] = playing_stat['HTFormPtsStr'].apply(self.detect)
        self.nr, self.part = 5, "WWWWW"
        playing_stat['HTWinStreak5'] = playing_stat['HTFormPtsStr'].apply(self.detect)
        self.nr, self.part = 3, "LLL"
        playing_stat['HTLossStreak3'] = playing_stat['HTFormPtsStr'].apply(self.detect)
        self.nr, self.part = 5, "LLLLL"
        playing_stat['HTLossStreak5'] = playing_stat['HTFormPtsStr'].apply(self.detect)
        self.nr, self.part = 3, "WWW"
        playing_stat['ATWinStreak3'] = playing_stat['ATFormPtsStr'].apply(self.detect)
        self.nr, self.part = 5, "WWWWW"
        playing_stat['ATWinStreak5'] = playing_stat['ATFormPtsStr'].apply(self.detect)
        self.nr, self.part = 3, "LLL"
        playing_stat['ATLossStreak3'] = playing_stat['ATFormPtsStr'].apply(self.detect)
        self.nr, self.part = 5, "LLLLL"
        playing_stat['ATLossStreak5'] = playing_stat['ATFormPtsStr'].apply(self.detect)

        # Get Goal Difference
        playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
        playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']

        # Diff in points
        playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
        playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']

        # Diff in last year positions
        playing_stat['DiffLP'] = playing_stat['HomeTeamLP'] - playing_stat['AwayTeamLP']

        # Scale DiffPts, DiffFormPts, HTGD, ATGD by Matchweek.
        cols = ['HTGD','ATGD','DiffPts','DiffFormPts','HTP','ATP']
        playing_stat.MW = playing_stat.MW.astype(float)

        for col in cols:
            playing_stat[col] = playing_stat[col] / playing_stat.MW

        # make ftr binary
        playing_stat['FTR'] = playing_stat.FTR.apply(self.binary)

        playing_stat.to_csv(self.path + "final/final.csv")

if __name__ == '__main__':
    f = Filter()
    f.main()
