import pandas as pd
import xgboost as xgb
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
pd.options.mode.chained_assignment = None
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def ftr_score(string):
    if string == 'H':
        return 0
    elif string == "A":
        return 1
    else:
        return 2


def last5_score(string):
    if string == 'N':
        return 0
    elif string == "W":
        return 1
    else:
        return 2


def preprocess_features(data):
    """
    Preprocesses football data and
    converts categorical variables into dummy variables.
    """
    # new dataframe
    output = pd.DataFrame(index=data.index)

    # iterate each column and find what type of data is in there
    for col, col_data in data.iteritems():

        # if data type is categorical (for example team names),
        # convert to dummy/indicator variables (0's and 1's)
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)

        # add columns to output
        output = output.join(col_data)
    return output


def change(data):

    # change FTR letters to numbers
    data['FTR'] = data.FTR.apply(ftr_score)

    # change last 5 matches
    data['HM1'] = data.HM1.apply(last5_score)
    data['HM2'] = data.HM2.apply(last5_score)
    data['HM3'] = data.HM3.apply(last5_score)
    data['HM4'] = data.HM4.apply(last5_score)
    data['HM5'] = data.HM5.apply(last5_score)

    data['AM1'] = data.AM1.apply(last5_score)
    data['AM2'] = data.AM2.apply(last5_score)
    data['AM3'] = data.AM3.apply(last5_score)
    data['AM4'] = data.AM4.apply(last5_score)
    data['AM5'] = data.AM5.apply(last5_score)
    return data


def train_classifier(clf, X_train, y_train):
    """
    Fits a classifier to the training data.
    """
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    print("Trained in: {:.4f} seconds.".format(end - start))


def predict_labels(clf, features, target):
    """
    Makes predictions using a fit classifier based on F1 score.
    """
    start = time()
    y_pred = clf.predict(features)
    end = time()

    print("Prediction: {:.4f} seconds.".format(end - start))

    return f1_score(target, y_pred,  average='weighted'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test, grid=False):
    """
    Train and predict using a classifer based on F1 score.
    """
    train_classifier(clf, X_train, y_train)
    if grid:
        clf = clf.best_estimator_
    f1, acc = predict_labels(clf, X_test, y_test)
    print("Test set f1 and acc: {:.4f} , {:.4f}.\n".format(f1, acc))
    return clf, f1, acc


class Trainer:

    # Read data
    data = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/data/final/final.csv")

    data = change(data)

    X_all = data.drop(['FTR', "FTHG", "FTAG", "Date", "HTFPS", "ATFPS"], axis=1)
    y_all = data['FTR']

    # remove categorical variables
    X_all = preprocess_features(X_all)

    # edit out this season
    this_season = X_all[:380]
    this_season.to_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/editedfuture.csv", index=False)

    # cut it out
    X_all = X_all[380:]
    y_all = y_all[380:]

    # print(X_all.head(5))

    # first 380 entries are last season
    # this is used to test the acc
    X_test = X_all[:380]
    y_test = y_all[:380]

    # rest can be shuffled
    X_rest = X_all[380:]
    y_rest = y_all[380:]


    def main(self):

        # check if has learned
        nr = self.y_test.shape[0]
        wins = len(self.y_test[self.y_test == 0])
        away = len(self.y_test[self.y_test == 1])
        draw = len(self.y_test[self.y_test == 2])
        print("Home win rate {:.4f}%".format(float(wins / nr) * 100))
        print("Away win rate {:.4f}%".format(float(away / nr) * 100))
        print("Draw rate {:.4f}%".format(float(draw / nr) * 100))
        print()

        best = 0.0
        best_seed = None
        model = None

        # change here
        boosters = ["gbtree"]
        colsample_bylevels = [0.87]  # list(np.arange(0.75, 0.9, 0.01))  # 0.8 - 0.9
        colsample_bynodes = [0.0]  # list(np.arange(0.0, 0.1, 0.01))  # 0.0 - 0.0
        colsample_bytrees = [0.69]  # list(np.arange(0.55, 0.7, 0.01))  # 0.6 - 0.65

        learning_rates = [0.32]  # list(np.arange(0.2, 0.4, 0.01))  # 0.3
        gammas = [0.3]  # list(np.arange(0.25, 0.35, 0.01))  # 0.3 0.4
        max_depths = [5]  # list(np.arange(4, 7, 1))  # 5

        min_child_weights = [1]  # list(np.arange(0, 3, 1))  # 1
        max_delta_steps = [0.0]  # list(np.arange(0.0, 0.1, 0.01))  # 0
        subsamples = [1.0]  # list(np.arange(0.9, 1.01, 0.01))  # 1.0

        # lambdas = list(np.arange(0, 1.1, 0.1))
        # alphas = [0]  # [0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
        # refresh_leafs = list(np.arange(0, 1.1, 0.1))
        # process_types = list(np.arange(0, 1.1, 0.1))
        # num_parallel_trees = list(np.arange(0, 10, 1))

        n_estimatorss = [101]  # list(np.arange(99, 104, 1))  # 103

        lista = [
            boosters,
            colsample_bylevels,
            colsample_bynodes,
            colsample_bytrees,
            learning_rates,
            gammas,
            max_depths,
            min_child_weights,
            max_delta_steps,
            subsamples,
            n_estimatorss
        ]
        combinations = list(itertools.product(*lista))
        print(len(combinations))

        n = 16

        start = time()
        for seed in range(n, n + 1):
            # make data
            # 1 row must be sacrificed, but that's okay
            X_train, _, y_train, _ = train_test_split(
                self.X_rest, self.y_rest,
                test_size=1,
                random_state=seed,
                shuffle=True,
                stratify=None
            )
            # clf_C = xgb.XGBClassifier(
            #     eval_metric='mlogloss',
            #     use_label_encoder=False
            # )
            for combination in combinations:
                print(combination)
                booster = combination[0]
                colsample_bylevel = combination[1]
                colsample_bynode = combination[2]
                colsample_bytree = combination[3]
                learning_rate = combination[4]
                gamma = combination[5]
                max_depth = combination[6]
                min_child_weight = combination[7]
                max_delta_step = combination[8]
                subsample = combination[9]
                n_estimators = combination[10]


                clf_base = xgb.XGBClassifier(
                    booster=booster,

                    colsample_bylevel=colsample_bylevel,
                    colsample_bynode=colsample_bynode,
                    colsample_bytree=colsample_bytree,

                    learning_rate=learning_rate,
                    gamma=gamma,
                    max_depth=max_depth,

                    min_child_weight=min_child_weight,
                    max_delta_step=max_delta_step,
                    subsample=subsample,

                    n_estimators=n_estimators,
                    # reg_alpha=reg_alpha,
                    # base_score=base_score,

                    # don't change
                    validate_parameters=False,
                    eval_metric='mlogloss',
                    num_class=3,
                    objective="multi:softmax",
                    use_label_encoder=False,
                    verbosity=1
                )
                # base
                clf, f1, acc = train_predict(clf_base, X_train, y_train, self.X_test, self.y_test)
                if acc > best:
                    best = acc
                    model = clf
                    best_seed = seed
            # # optimize
            # parameters = {
            #     'min_child_weight': [3],
            #     'gamma': [0.4],
            #     'subsample': [0.8],
            #     'colsample_bytree': [0.8],
            #     'max_depth': [3, 4],  # 3
            #     'learning_rate': [0.1],
            #     'n_estimators': [40],
            #     # 'scale_pos_weight' : [1],
            #     'reg_alpha': [1e-5],
            #     'booster': ["gbtree"]
            #     # "average": ['weighted']
            # }
            # clf = xgb.XGBClassifier(
            #     # booster="gbtree",  # gbtree, gblinear or dart
            #
            #     # dont change these
            #     validate_parameters=False,
            #     eval_metric='mlogloss',
            #     num_class=3,
            #     objective="multi:softmax",
            #     use_label_encoder=False
            # )
            #
            # f1_scorer = make_scorer(
            #     f1_score,
            #     average='weighted'
            # )
            #
            # grid_obj = GridSearchCV(
            #     clf,
            #     scoring=f1_scorer,
            #     param_grid=parameters,
            #     cv=5
            # )
            # clf, f1, acc = train_predict(grid_obj, X_train, y_train, self.X_test, self.y_test, grid=True)
            # if acc > best:
            #     best = acc
            #     model = clf
        end = time()
        print("Time taken: {:.4f} seconds.".format(end - start))
        name = str(int(best * 10000))
        print(model)
        print(name)
        print(best_seed)
        model.save_model(f"C:/Users/theerik/PycharmProjects/predictor/models/{name}.txt")


if __name__ == '__main__':
    t = Trainer()
    t.main()