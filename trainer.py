import pandas as pd
import xgboost as xgb
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
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

    X_all = data.drop(['FTR', "FTHG", "FTAG", "Date", "Unnamed: 0", "HTFPS", "ATFPS"], axis=1)
    y_all = data['FTR']

    # remove categorical variables
    X_all = preprocess_features(X_all)

    # edit out this season
    this_season = X_all[:380]
    this_season.to_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/editedfuture.csv", index=False)

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
        model = None

        # change here
        boosters = ["gbtree"]
        # none = 6
        max_depths = [6]  # best list(np.arange(1, 15, 1))
        # none = 1
        colsample_bytrees = [1]  # best
        # none = 0
        gammas = [0.1]  # list(np.arange(0.1, 1.1, 0.1)) #, 0.8, 0.4]
        # none = 1
        subsamples = [1]  # list(np.arange(0.1, 1.1, 0.1))  #, 0.8, 0.6]
        min_child_weights = [10]  # best
        learning_rates = [0.16]  # best
        n_estimatorss = [10]  # best

        # list(np.arange(start, stop, step))
        reg_alphas = [0.00001]  # sort

        # Bad design but its the easiest
        for seed in range(1, 2):
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
            # todo filter if there is "illegal" combination and dont calculate it twice
            for booster in boosters:
                for max_depth in max_depths:
                    for gamma in gammas:
                        for colsample_bytree in colsample_bytrees:
                            for subsample in subsamples:
                                for min_child_weight in min_child_weights:
                                    for learning_rate in learning_rates:
                                        for n_estimators in n_estimatorss:
                                            for reg_alpha in reg_alphas:
                                                values1 = [max_depth, gamma, colsample_bytree, subsample]
                                                print(booster, max_depth, gamma, colsample_bytree, subsample,
                                                      min_child_weight, learning_rate, n_estimators, reg_alpha)
                                                if booster == "gblinear" and not all(v is None for v in values1):
                                                    continue

                                                clf_C = xgb.XGBClassifier(
                                                    booster=booster,

                                                    gamma=gamma,
                                                    subsample=subsample,
                                                    colsample_bytree=colsample_bytree,
                                                    max_depth=max_depth,

                                                    min_child_weight=min_child_weight,
                                                    learning_rate=learning_rate,
                                                    n_estimators=n_estimators,
                                                    reg_alpha=reg_alpha,

                                                    # don't change
                                                    validate_parameters=False,
                                                    eval_metric='mlogloss',
                                                    num_class=3,
                                                    objective="multi:softmax",
                                                    use_label_encoder=False,
                                                    verbosity=1
                                                )

                                                # base
                                                clf, f1, acc = train_predict(clf_C, X_train, y_train, self.X_test, self.y_test)
                                                if acc > best:
                                                    best = acc
                                                    model = clf
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
        name = str(int(best * 10000))
        print(model)
        print(name)
        model.save_model(f"C:/Users/theerik/PycharmProjects/predictor/models/{name}.txt")


if __name__ == '__main__':
    t = Trainer()
    t.main()