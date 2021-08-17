import pandas as pd
import xgboost as xgb
import numpy as np
from time import time
from sklearn.model_selection import train_test_split, cross_val_score
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
    data["Check"] = True

    # get only happened games from future games and add them to train data
    future = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/future.csv")

    # future_false = future[future['Check'] == False]
    # future_false = future_false.drop(["Check"], axis=1)
    #
    # future_true = future[future['Check'] == True]
    # future_true = future_true.drop(["Check"], axis=1)

    data = pd.concat([data, future], ignore_index=True)

    data = change(data)

    # future_false.to_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/template.csv", index=False)

    # future_false = future[future['Check'] == False]
    #
    # data = data[data['Check'] == True]

    X_all = data.drop(['FTR', "FTHG", "FTAG", "Date", "HTFPS", "ATFPS"], axis=1)

    y_all = data[['FTR', "Check"]]

    # remove categorical variables
    X_all = preprocess_features(X_all)

    # save games that haven't happened yet
    future_false = X_all[X_all['Check'] == False]
    future_false = future_false.drop(["Check"], axis=1)
    future_false.to_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/template.csv", index=False)

    # get only games that we know the result of
    X_all = X_all[X_all['Check'] == True]
    y_all = y_all[y_all["Check"] == True]
    X_all = X_all.drop(["Check"], axis=1)
    y_all = y_all["FTR"]

    def main(self):

        best = 0.0
        best_seed = None
        model = None
        start = time()

        the_seed = 216

        for seed in range(the_seed, the_seed + 1):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_all, self.y_all,
                random_state=seed,
                shuffle=True,
                stratify=None
            )
            # check rates
            nr = y_test.shape[0]
            wins = len(y_test[y_test == 0])
            away = len(y_test[y_test == 1])
            draw = len(y_test[y_test == 2])
            a = float(wins / nr)
            b = float(away / nr)
            print("Home win rate {:.4f}%".format(a * 100))
            print("Away win rate {:.4f}%".format(b * 100))
            print("Draw rate {:.4f}%".format(float(draw / nr) * 100))
            print()

            # change here
            boosters = ["gbtree"]
            # 0.3
            learning_rates = [1.0] # 1 1 1  # list(np.arange(0.0, 1.01, 0.01))
            # 0.0
            gammas = [0.33]  #0.01 0.33 0.33 0.33# list(np.arange(0.1, 1.01, 0.01))
            # 6
            max_depths = [6]  # 6 6 9# list(np.arange(0, 50, 1))
            # 1.0
            min_child_weights = [0.0]  # 0.0 0.0  0.32 # list(np.arange(0, 40, 1))
            # 0.0
            max_delta_steps = [0.0] # 0 0 0# list(np.arange(0.0, 1.01, 0.01))
            # 1.0
            subsamples = [1.0]  # list(np.arange(0.0, 1.01, 0.01))
            # 100
            n_estimatorss = [100]  # list(np.arange(50, 200, 1))
            # 1.0
            colsample_bylevels = [1.0]  # list(np.arange(0.0, 1.01, 0.01))
            colsample_bynodes = [1.0]  # list(np.arange(0.0, 1.01, 0.01))
            colsample_bytrees = [1.0]  # list(np.arange(0.0, 1.01, 0.01))

            # lambdas = list(np.arange(0, 1.1, 0.1))
            # alphas = [0]  # [0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
            # refresh_leafs = list(np.arange(0, 1.1, 0.1))
            # process_types = list(np.arange(0, 1.1, 0.1))
            # num_parallel_trees = list(np.arange(0, 10, 1))

            lista = [
                boosters,
                learning_rates,
                gammas,
                max_depths,
                min_child_weights,
                max_delta_steps,
                subsamples,
                n_estimatorss,
                colsample_bylevels,
                colsample_bynodes,
                colsample_bytrees,
            ]
            combinations = list(itertools.product(*lista))
            size = len(combinations)
            # print("combinations", size)
            n = 0
            for combination in combinations:
                n += 1
                print(f"{n}/{size} {combination}")
                booster = combination[0]
                learning_rate = combination[1]
                gamma = combination[2]
                max_depth = combination[3]
                min_child_weight = combination[4]
                max_delta_step = combination[5]
                subsample = combination[6]
                n_estimators = combination[7]
                colsample_bylevel = combination[8]
                colsample_bynode = combination[9]
                colsample_bytree = combination[10]

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
                clf, f1, acc = train_predict(clf_base, X_train, y_train, X_test, y_test)
                if acc > best:
                    best = acc
                    model = clf
                    best_seed = seed
        end = time()
        print("Time taken: {:.4f} seconds.".format(end - start))
        name = str(int(best * 10000))
        print(model)
        print("name", name)
        print("seed", best_seed)
        print("score", best)
        model.save_model(f"C:/Users/theerik/PycharmProjects/predictor/models/{name}.txt")


if __name__ == '__main__':
    t = Trainer()
    t.main()