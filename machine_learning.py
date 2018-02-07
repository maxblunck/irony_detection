from feature_extraction import extract_features
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn import metrics
import config
import csv
import utils


def run(corpus):
    """
    Main part of the programm. Includes following steps:
        - split the data
        - extract features
        - initiate classifiers
        - train test, validate & tune classifiers
        - evaluate results
    """

    print("----------Configuration----------\nSelected Features: {}\n".format(config.feature_selection))

    # split data set according to config
    split_point = int(config.split_ratio*len(corpus))
    train_set = corpus[:split_point]
    test_set = corpus[split_point:]

    # inputs (x)
    train_inputs, test_inputs, features = extract_features(train_set, test_set)

    # labels (y)
    train_labels = np.array([int(el['LABEL']) for el in train_set])  # 1000 labels
    test_labels = np.array([int(el['LABEL']) for el in test_set])  # 254 labels

    # initiate classifiers
    svm_clf = svm.SVC(C=0.1, gamma=0.001, kernel='linear') # best: C=0.1, gamma=0.001, kernel='linear'
    tree_clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=2) # best: criterion='gini', max_depth=6, min_samples_leaf=2
    nb_clf = naive_bayes.MultinomialNB(alpha=0.5) # best: alpha=0.5
    lr_clf = linear_model.LogisticRegression(C=0.1) # best: C=0.1 #, 'penalty': 'l2'

    # validation
    if config.validate == True:
        validate_multiple([svm_clf, tree_clf, nb_clf, lr_clf], train_inputs, train_labels, 10) 

    # tuning (takes ~24h!)
    if config.tune == True:
        tune_multiple(svm_clf, tree_clf, nb_clf, lr_clf, train_inputs, train_labels)

    # training
    if config.train or config.test or config.plot_weights or config.log_misclassifications:
        train_multiple([svm_clf, tree_clf, nb_clf, lr_clf], train_inputs, train_labels)

    # testing
    if config.test == True:
        test_multiple([svm_clf, tree_clf, nb_clf, lr_clf], test_inputs, test_labels)

    # misclassification (SVM)
    if config.log_misclassifications == True:
        utils.get_misclassifications(svm_clf, test_set, test_inputs, test_labels)

    # plot top features (SVM)
    if config.plot_weights == True:
        feature_names = utils.get_all_feature_names(features)
        utils.plot_coefficients(svm_clf, feature_names)


def train_multiple(classifiers, train_inputs, train_labels):
    for classifier in classifiers:
        classifier.fit(train_inputs, train_labels)


def validate_multiple(classifiers, train_inputs, train_labels, folds):
    """
    Runs cross validation for given classifiers.
    Prints and writes results to file given in config.
    """
    print("\n-------Cross Validation-------")

    csv_file = open(config.validation_result_out, "a")
    csv_writer = csv.DictWriter(csv_file, ["classifier", "features", "accuracy", "precision", "recall", "f1-score"])

    for classifier in classifiers:

        scoring = ['accuracy', 'precision', 'recall', 'f1']
        scores = cross_validate(classifier, train_inputs, train_labels, scoring=scoring, 
                                cv=folds, return_train_score=False)

        acc = float("{0:.3f}".format(scores['test_accuracy'].mean()))
        prec = float("{0:.3f}".format(scores['test_precision'].mean()))
        recall = float("{0:.3f}".format(scores['test_recall'].mean()))
        f1 = float("{0:.3f}".format(scores['test_f1'].mean()))

        name = classifier.__str__().split("(")[0]
        
        print("{}\nAccuracy: {}, Recall: {}, Precision: {}, F1-Score: {}\n".format(name, acc, recall, prec, f1))

        csv_writer.writerow({"classifier":name, "features":config.feature_selection, "accuracy":acc,
                             "precision":prec, "recall":recall, "f1-score":f1
                             })
    csv_file.close()


def get_best_params(classifier, param_grid, train_inputs, train_labels):
    """
    Performs GridSearch to find best parameters
    """
    print("{} \n".format(classifier))

    grid_search = GridSearchCV(classifier, param_grid, cv=3)
    grid_search.fit(train_inputs, train_labels)

    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best score: {}".format(grid_search.best_score_))


def tune_multiple(svm, tree, nb, lr, train_inputs, train_labels):
    svm_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
                      'gamma' : [0.001, 0.01, 0.1, 1],
                      'kernel' : ['linear', 'rbf', 'poly']}

    tree_param_grid = {'criterion' : ['gini', 'entropy'],
                       'max_depth': [9, 6, 3, None],
                       'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 500],
                       'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9]}

    nb_param_grid = {'alpha' : [0, 0.5, 1.0]}

    lr_param_grid = {'penalty' : ['l1', 'l2'],
                     'C' : [0.001, 0.01, 0.1, 1, 10]}

    # TODO: something better
    get_best_params(svm, svm_param_grid, train_inputs, train_labels)
    get_best_params(tree, tree_param_grid, train_inputs, train_labels)
    get_best_params(nb, nb_param_grid, train_inputs, train_labels)
    get_best_params(lr, lr_param_grid, train_inputs, train_labels)


def test_multiple(classifiers, test_inputs, test_labels):
    """
    Evaluates given classifiers on test data.
    Prints and writes results to file given in config.
    """

    print("\n--------Test Data Scores----------")

    csv_file = open(config.test_result_out, "a")
    csv_writer = csv.DictWriter(csv_file, ["classifier", "features", "accuracy", "precision", "recall", "f1-score"])

    for clf in classifiers:
        
        acc = float("{0:.3f}".format(clf.score(test_inputs, test_labels)))
        predictions = clf.predict(test_inputs)
        f1 = float("{0:.3f}".format(metrics.f1_score(test_labels, predictions)))
        recall = float("{0:.3f}".format(metrics.recall_score(test_labels, predictions)))
        precision = float("{0:.3f}".format(metrics.precision_score(test_labels, predictions)))
        name = clf.__str__().split("(")[0]
        
        print("{}\nAccuracy: {}, Recall: {}, Precision: {}, F1-Score: {}\n".format(name, acc, recall, precision, f1))

        csv_writer.writerow({"classifier":name, "features":config.feature_selection, "accuracy":acc,
                             "precision":precision, "recall":recall, "f1-score":f1
                             })
    csv_file.close()


