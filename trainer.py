import pandas as pd
import xgboost as xgb
import numpy as np
from time import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

def ftr_score_home_or_not(string):
    if string == 'H':
        return 0
    else:
        return 1

def ftr_score_away_or_not(string):
    if string == 'A':
        return 0
    else:
        return 1


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


# def change(data):
#
#     # change FTR letters to numbers
#     data['FTRH'] = data.FTR.apply(ftr_score_home_or_not)
#     data['FTRA'] = data.FTR.apply(ftr_score_away_or_not)
#     data['FTR'] = data.FTR.apply(ftr_score)
#
#     # change last 5 matches
#     data['HM1'] = data.HM1.apply(last5_score)
#     data['HM2'] = data.HM2.apply(last5_score)
#     data['HM3'] = data.HM3.apply(last5_score)
#     data['HM4'] = data.HM4.apply(last5_score)
#     data['HM5'] = data.HM5.apply(last5_score)
#
#     data['AM1'] = data.AM1.apply(last5_score)
#     data['AM2'] = data.AM2.apply(last5_score)
#     data['AM3'] = data.AM3.apply(last5_score)
#     data['AM4'] = data.AM4.apply(last5_score)
#     data['AM5'] = data.AM5.apply(last5_score)
#     return data


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
    # if use grid search
    if grid:
        clf = clf.best_estimator_
    f1, acc= predict_labels(clf, X_test, y_test)
    print("Test set f1 and acc: {:.4f} , {:.4f}.\n".format(f1, acc))
    return clf, f1, acc

def tlist(name, lista):
    return [(name, x) for x in lista]


class Trainer:

    # Read data
    data = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/data/final/final.csv")
    data["Check"] = True
    future = pd.read_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/future.csv")

    # this number should be the same as happend games nr
    # print(len(future[future["Check"] == True]))

    data = pd.concat([data, future], ignore_index=True)

    X_all = data.drop(['FTR', "FTHG", "FTAG", "FTRH", "FTRA", "Date", "HTFPS", "ATFPS"], axis=1)

    y_all = data[['FTR', "FTRH", "FTRA", "Check"]]

    # remove categorical variables
    X_all = preprocess_features(X_all)

    # save games that haven't happened yet
    future_false = X_all[X_all['Check'] == False]

    future_false = future_false.drop(["Check"], axis=1)
    future_false.to_csv("C:/Users/theerik/PycharmProjects/predictor/data/futureGames/template.csv", index=False)

    # get only games that we know the result of
    X_all = X_all[X_all['Check'] == True]
    y_all = y_all[y_all["Check"] == True]

    # edit data smaller
    X_all = X_all.drop(["Check"], axis=1)
    y_all = y_all.drop(["Check"], axis=1)

    def trainer(self, y_part, seed, combinations_list, model_type):

        best_acc = 0.0
        best_seed = None
        model = None
        start = time()
        val_list = []

        # spilt data here so best seed can be found
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_all, self.y_all[y_part],
            random_state=seed,
            shuffle=True,
            stratify=None
        )
        # show rates
        nr = self.y_all.shape[0]
        a = len(self.y_all[self.y_all[y_part] == 0])
        b = len(self.y_all[self.y_all[y_part] == 1])
        c = len(self.y_all[self.y_all[y_part] == 2])
        print("A rate {:.4f}%".format(float(a / nr) * 100))
        print("B rate {:.4f}%".format(float(b / nr) * 100))
        print("C rate {:.4f}%".format(float(c / nr) * 100))

        # go through all the diff combinations
        combinations = list(itertools.product(*combinations_list))
        size = len(combinations)
        n = 0
        for combination in combinations:
            n += 1
            print(f"{n}/{size} {combination}")
            config = {}
            for i in combination:
                config[i[0]] = i[1]
            clf_base = model_type(**config)
            clf, f1, acc = train_predict(clf_base, X_train, y_train, X_test, y_test)
            if acc > best_acc:
                best_acc = acc
                model = clf
                best_seed = seed
                val_list = [config]
            elif acc == best_acc:
                val_list.append(config)
        end = time()
        print("Time taken: {:.4f} seconds.".format(end - start))
        name = str(int(best_acc * 10000))
        print(model)
        print("name", name)
        print("seed", best_seed)
        print("score", best_acc)
        print("list size", len(val_list))
        for i in val_list:
            print(i)
        return model, name

    def main(self):
        # change here
        boosters = ["gbtree"]
        # 100
        n_estimatorss = [5]  # list(np.arange(0, 150, 1))
        # 0.3
        learning_rates = [0.14]  #list(np.arange(0.0, 1.01, 0.01))
        # 0.0
        gammas = [0.93, 0.94]  # list(np.arange(0.0, 1.01, 0.01))
        # 6
        max_depths = [2]  # list(np.arange(1, 20, 1))
        # 1
        min_child_weights = [17, 18, 19]  # list(np.arange(1, 20, 1))
        # 0.0
        max_delta_steps = []  # list(np.arange(0.0, 1.01, 0.01))
        # 1.0
        subsamples = [1.0]  # list(np.arange(0.0, 1.01, 0.01))
        # 1.0
        colsample_bylevels = [1.0]  # list(np.arange(0.0, 1.01, 0.01))
        colsample_bynodes = [1.0]  # list(np.arange(0.0, 1.01, 0.01))
        colsample_bytrees = [1.0]  # list(np.arange(0.0, 1.01, 0.01))
        # 0.0
        lambdas = [0.0]  # list(np.arange(0.0, 1.01, 0.01))
        # 1.0
        alphas = [1.0]  # list(np.arange(0.0, 1.01, 0.01))

        lista = [
            tlist("booster", boosters),
            tlist("n_estimators", n_estimatorss),
            tlist("learning_rate", learning_rates),
            tlist("gamma", gammas),
            tlist("max_depth", max_depths),
            tlist("min_child_weight", min_child_weights),
            tlist("max_delta_step", max_delta_steps),
            tlist("subsample", subsamples),
            tlist("colsample_bylevel", colsample_bylevels),
            tlist("colsample_bynode", colsample_bynodes),
            tlist("colsample_bytree", colsample_bytrees),
            tlist("lambda", lambdas),
            tlist("alpha", alphas),

            # dont change
            tlist("validate_parameters", [False]),
            tlist("eval_metric", ['mlogloss']),
            tlist("num_class", [3]),
            tlist("objective", ["multi:softmax"]),
            tlist("use_label_encoder", [False]),
            tlist("verbosity", [1]),
        ]

        model, name = self.trainer(
            y_part="FTR",
            seed=0,
            combinations_list=lista,
            model_type=xgb.XGBClassifier
        )
        model.save_model(f"C:/Users/theerik/PycharmProjects/predictor/models/{name}.txt")

    def binary_main(self):
        """
        same as main but use binary classification
        """
        ftr_type = "FTRH"

        # change here
        boosters = ["gbtree"]
        # 100
        n_estimatorss = [86]  # list(np.arange(60, 120, 1))
        # 0.3
        learning_rates = [0.549]  # list(np.arange(0.0, 1.001, 0.001))
        # 0.0
        gammas = [0.496]  # list(np.arange(0.0, 1.001, 0.001))
        # 6
        max_depths = [6]  # list(np.arange(0, 30, 1))
        # 1
        min_child_weights = [1]  # list(np.arange(0, 20, 1))
        # 0.0
        max_delta_steps = [0.12]  # list(np.arange(0.0, 1.001, 0.001))
        # 1.0
        subsamples = [0.154]  # list(np.arange(0.0, 1.001, 0.001))
        # 1.0
        colsample_bylevels = [1.0]  # list(np.arange(0.0, 1.001, 0.001))
        colsample_bynodes = [1.0]  # list(np.arange(0.0, 1.001, 0.001))
        colsample_bytrees = [1.0]  # list(np.arange(0.0, 1.001, 0.001))
        # 0.0
        lambdas = [0.986]  # list(np.arange(0.0, 1.001, 0.001))
        # 1.0
        alphas = [0.991]  # list(np.arange(0.0, 1.001, 0.001))

        lista = [
            tlist("booster", boosters),
            tlist("n_estimators", n_estimatorss),
            tlist("learning_rate", learning_rates),
            tlist("gamma", gammas),
            tlist("max_depth", max_depths),
            tlist("min_child_weight", min_child_weights),
            tlist("max_delta_step", max_delta_steps),
            tlist("subsample", subsamples),
            tlist("colsample_bylevel", colsample_bylevels),
            tlist("colsample_bynode", colsample_bynodes),
            tlist("colsample_bytree", colsample_bytrees),
            tlist("lambda", lambdas),
            tlist("alpha", alphas),

            # dont change
            tlist("validate_parameters", [False]),
            tlist("eval_metric", ['auc']),
            # tlist("num_class", [2]),
            tlist("objective", ["binary:logistic"]),
            tlist("use_label_encoder", [False]),
            tlist("verbosity", [1]),
        ]

        model, name = self.trainer(
            y_part=ftr_type,
            seed=1,
            combinations_list=lista,
            model_type=xgb.XGBClassifier
        )

        model.save_model(f"C:/Users/theerik/PycharmProjects/predictor/models/b{name}.txt")




if __name__ == '__main__':
    t = Trainer()
    t.main()
    # t.binary_main()

