import corpus
import features
import surface_patterns
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
import time
import pickle
import config
import csv
import itertools as it


def train_multiple(classifiers, train_inputs, train_labels):
    for classifier in classifiers:
        classifier.fit(train_inputs, train_labels)


def validate_multiple(classifiers, train_inputs, train_labels):
    print("\n-------Cross Validation-------")

    for classifier in classifiers:
        print("\n{}".format(classifier))

        accuracy = cross_val_score(classifier, train_inputs, train_labels, cv=3, scoring='accuracy').mean()
        f1 = cross_val_score(classifier, train_inputs, train_labels, cv=3, scoring='f1').mean()
        
        print("\nAccuracy: {}, F1-Score: {}\n".format(accuracy, f1))


def get_best_params(classifier, param_grid, train_inputs, train_labels):

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

    # TODO: irgendwas
    get_best_params(svm, svm_param_grid, train_inputs, train_labels)
    get_best_params(tree, tree_param_grid, train_inputs, train_labels)
    get_best_params(nb, nb_param_grid, train_inputs, train_labels)
    get_best_params(lr, lr_param_grid, train_inputs, train_labels)


def test_multiple(classifiers, test_inputs, test_labels, csv_writer=None):
    print("\n--------Test Data Scores----------")

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


def main():
    print("----------Configuration----------\nFeatures: {}\n".format(config.feature_selection))
    amz_corpus = corpus.read_corpus("corpus_shuffled.csv")
    extended_corpus = surface_patterns.extract_surface_patterns(amz_corpus, 1000)

    # split data set 80:20
    train_set = extended_corpus[:1000]
    test_set = extended_corpus[1000:]

    train_inputs, train_labels = [], []
    test_inputs, test_labels = [], []

    if config.re_extract == True:

        # inputs (x)
        train_inputs, test_inputs = features.extract_features(train_set, test_set)
        # print("\n---> Duration Feature Extraction: {} sec.\n".format(int(time.time()-start_time)))

        # labels (y)
        train_labels = np.array([int(el['LABEL']) for el in train_set])  # 1000 labels
        test_labels = np.array([int(el['LABEL']) for el in test_set])  # 254 labels

        # TODO: save to pickle
        #pickle.dump([train_inputs, train_labels, test_inputs, test_labels], open("vectors.pickle", "wb"))

    else:
        # load from pickle
        v = pickle.load(open("vectors.pickle", "rb"))
        train_inputs, train_labels = v[0], v[1]
        test_inputs, test_labels = v[2], v[3]


    # Machine Learning

    # init
    svm_clf = svm.SVC(C=0.1, gamma=0.001, kernel='linear') # best: C=0.1, gamma=0.001, kernel='linear'
    tree_clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=2) # best: 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 2
    nb_clf = naive_bayes.MultinomialNB(alpha=0.5) # best: 'alpha': 0.5
    lr_clf = linear_model.LogisticRegression(C=0.01) # best: 'C': 0.01, 'penalty': 'l2'

    # validation
    if config.validate == True:
        validate_multiple([svm_clf], train_inputs, train_labels) #, tree_clf, nb_clf, lr_clf
        #print("---> Duration CV: {} sec.".format(int(time.time()-start_time)))

    # tuning (takes ~24h!)
    if config.tune == True:
        tune_multiple(svm_clf, tree_clf, nb_clf, lr_clf, train_inputs, train_labels)
        #print("---> Duration param search: {} sec.".format(int(time.time()-start_time)))

    # training
    if config.train == True:
        train_multiple([svm_clf, tree_clf, nb_clf, lr_clf], train_inputs, train_labels)
        #print("---> Duration Training: {} sec.\n".format(int(time.time()-start_time)))

    # testing
    if config.test == True:
        csv_file = open(config.test_result_out, "a")
        writer = csv.DictWriter(csv_file, ["classifier", "features", "accuracy", "precision", "recall", "f1-score"])
        test_multiple([svm_clf, tree_clf, nb_clf, lr_clf], test_inputs, test_labels, writer)
        csv_file.close()

    #TODO save best model to pickle


if __name__ == '__main__':
    start_time = time.time()

    csv_file = open(config.test_result_out, "w")
    writer = csv.DictWriter(csv_file, ["classifier", "features", "accuracy", "precision", "recall", "f1-score"])
    writer.writeheader()
    csv_file.close()

    if config.use_all_variants == True:
        
        f_combinations = []

        # add all possible feature-combos to array
        for i in range(1, len(config.feature_selection)+1):
            combination = it.combinations(config.feature_selection, i)
            for el in combination:
                f_combinations.append(list(el))

        # run main for all feature combos
        count = 1
        for combo in f_combinations:
            config.feature_selection = combo

            print("\nRunning configuration {} of {}".format(count, len(f_combinations)))
            count += 1
    
            main()

    else:
        main()



