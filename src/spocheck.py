#!/usr/bin/env python3

"""
If you use my code or material in your own project, please reference the source, including:

The Name of the author, e.g. “Jason Brownlee”.
The Title of the tutorial or book.
The Name of the website, e.g. “Machine Learning Mastery”.
The URL of the tutorial or book.
The Date you accessed or copied the code.
For example:

Jason Brownlee, Machine Learning Algorithms in Python, Machine Learning Mastery,
Available from https://machinelearningmastery.com/machine-learning-with-python/, accessed April 15th, 2018.
Also, if your work is public, contact me, I’d love to see it out of general interest.
https://machinelearningmastery.com/spot-check-machine-learning-algorithms-in-python/
"""

import warnings
from numpy import mean
from numpy import std
from matplotlib import pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg
from datasets.radioml import class_to_index
from utils import get_repo_path

from sklearn import model_selection
from sklearn.metrics import classification_report
# TODO: from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


def spot_check_cross_validation(X, y, k_fold=10, test_size=0.10,
                                random_state=None):
    scoring = 'accuracy'
    n_splits = k_fold
    # seed = 7
    X_train, X_test, y_train, y_test = \
        model_selection.train_test_split(X, y, test_size=test_size)
    models = get_models()

    results = []
    kfold = model_selection.KFold(n_splits=n_splits, shuffle=True,
                                  random_state=random_state)

    for name, model in models.items():
        cv_results = model_selection.cross_val_score(model, X_train, y_train,
                                                     cv=kfold, scoring=scoring)
        results.append((name, cv_results))

        msg = "\n\n%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        msg += "\n" + str(cv_results)
        print(msg)
        print("\n")
        print("Make predictions on test df for " + name)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("accuracy_score = " + str(accuracy_score(y_test, predictions)))
        # print(confusion_matrix(y_test, predictions))
        # print(classification_report(y_test, predictions))


def spot_check(X, y, k_fold=None, test_size=0.10, seed=None):
    X = pd.DataFrame(preprocessing.scale(X))
    if k_fold is not None:
        spot_check_cross_validation(X, y, k_fold=k_fold, test_size=test_size)
        return
    # scoring = 'accuracy'
    X_train, X_test, y_train, y_test = model_selection\
        .train_test_split(X, y, test_size=test_size, random_state=seed)

    models = get_models()
    # results = []

    for name, model in models.items():
        print("Make predictions on test df for " + name)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        print(accuracy_score(y_test, predictions))
        print(cm)
        print(classification_report(y_test, predictions))
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
        #                               display_labels=y_test.unique())
        # disp.plot()
        plot_confusion_matrix(model, X_test, y_test)
        plt.show()

    return models


# create a dict of standard models to evaluate {name:object}
def get_models():
    models = dict()

    # linear models
    models['logistic'] = LogisticRegression()
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for a in alpha:
        models['ridge-' + str(a)] = RidgeClassifier(alpha=a)
    models['sgd'] = SGDClassifier(max_iter=1000, tol=1e-3)
    models['pa'] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
    models['lda'] = LinearDiscriminantAnalysis()

    # non-linear models
    models['cart'] = DecisionTreeClassifier()
    models['extra'] = ExtraTreeClassifier()
    models['svml'] = SVC(kernel='linear')
    models['svmp'] = SVC(kernel='poly')
    models['bayes'] = GaussianNB()

    n_neighbors = range(1, 21)
    # n_neighbors = [2, 6, 21]

    for k in n_neighbors:
        models['knn-' + str(k)] = KNeighborsClassifier(n_neighbors=k)

    c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # c_values = [0.1,  0.6,  1.0]

    for c in c_values:
        models['svmr' + str(c)] = SVC(C=c)

    # ensemble models
    n_trees = 100
    models['ada'] = AdaBoostClassifier(n_estimators=n_trees)
    models['bag'] = BaggingClassifier(n_estimators=n_trees)
    models['rf'] = RandomForestClassifier(n_estimators=n_trees)
    models['et'] = ExtraTreesClassifier(n_estimators=n_trees)
    models['gbm'] = GradientBoostingClassifier(n_estimators=n_trees)

    print('Defined %d models' % len(models))
    return models


# create a feature preparation pipeline for a model
def make_pipeline(model):
    steps = list()
    # standardization
    steps.append(('standardize', StandardScaler()))
    # normalization
    steps.append(('normalize', MinMaxScaler()))
    # the model
    steps.append(('model', model))
    # create pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline


# evaluate a single model
def evaluate_model(X, y, model, folds, metric):
    # create the pipeline
    pipeline = make_pipeline(model)
    # evaluate model
    scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1)
    return scores


# evaluate a model and try to trap errors and and hide warnings
def robust_evaluate_model(X, y, model, folds, metric):
    # scores = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = evaluate_model(X, y, model, folds, metric)
    except:
        scores = None
    return scores


# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(X, y, models, folds=10, metric='accuracy'):
    results = dict()
    for name, model in models.items():
        # evaluate the model
        scores = robust_evaluate_model(X, y, model, folds, metric)
        # show process
        if scores is not None:
            # store a result
            results[name] = scores
            mean_score, std_score = mean(scores), std(scores)
            print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))
        else:
            print('>%s: error' % name)
    return results


# print and plot the top n results
def summarize_results(results, maximize=True, top_n=10):
    # check for no results
    if not results:
        print('no results')
        return
    # determine how many results to summarize
    n = min(top_n, len(results))
    # create a list of (name, mean(scores)) tuples
    mean_scores = [(k, mean(v)) for k, v in results.items()]
    # sort tuples by mean score
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
    # reverse for descending order (e.g. for accuracy)
    if maximize:
        mean_scores = list(reversed(mean_scores))
    # retrieve the top n for summarization
    names = [x[0] for x in mean_scores[:n]]
    scores = [results[x[0]] for x in mean_scores[:n]]
    # print the top n
    print()
    for i, name in enumerate(names):
        mean_score, std_score = mean(results[name]), std(results[name])
        print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i+1, name, mean_score, std_score))
    # boxplot for the top n
    pyplot.boxplot(scores, labels=names)
    _, labels = pyplot.xticks()
    pyplot.setp(labels, rotation=90)
    pyplot.savefig('spotcheck.png')


# load the dataset, returns X and y elements
def load_dataset():

    df = pd.read_pickle(get_repo_path() / cfg.Spotcheck.input_file)

    y = df.index.get_level_values('modulation_type').map(class_to_index)
    X = np.array(df)

    return X, y


    return make_classification(n_samples=1000, n_classes=2, random_state=1)


def main():
    # load dataset
    X, y = load_dataset()
    # get model list
    models = get_models()
    # evaluate models
    results = evaluate_models(X, y, models)
    # summarize results
    summarize_results(results)


if __name__ == '__main__':
    main()
