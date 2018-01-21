import corpus
import sent_rating_feature
import ngram_feature
import pos_feature
import punctuation_feature
import surface_patterns
import contrast_feature
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, GridSearchCV
import time
import pickle


def extract_features(training_set, test_set):

    # vocabularies
    n_gram_vocab = ngram_feature.get_vocabulary(train_set, 'REVIEW', (1,1)) # n1==n2!
    pos_bigram_vocab = pos_feature.get_pos_vocabulary(train_set)
    surface_bigram_vocab = ngram_feature.get_vocabulary(train_set, 'SURFACE_PATTERNS', (3,3))

    # inputs:
    print("------Feature Extraction------\n")
    train_inputs = [create_vector(el, n_gram_vocab, pos_bigram_vocab, surface_bigram_vocab)
                    for el in train_set]  # 1000 vectors
    test_inputs = [create_vector(el, n_gram_vocab, pos_bigram_vocab, surface_bigram_vocab)
                   for el in test_set]  # 254 vectors

    # stats
    print("Number of train samples:          {}".format(len(train_inputs)))
    print("N_gram vocab size:                {}".format(len(n_gram_vocab)))
    print("POS-Bigram vocab size:            {}".format(len(pos_bigram_vocab)))
    print("SP-Bigram vocab size:             {}".format(len(surface_bigram_vocab)))
    print("Total features per train sample:  {}".format(len(train_inputs[0])))
    print("---> Duration Feature Extraction: {} sec.\n".format(int(time.time()-start_time)))

    return train_inputs, test_inputs


def create_vector(corpus_instance, vocabulary=None, pos_vocabulary=None, surface_vocabulary=None):
    """
    Calls all feature extraction programms and combines
    resulting arrays to a single input vector (for a
    single corpus instance)
    Example for corpus instance: OrderedDict([('LABEL', '0'), ('STARS', '5.0'), etc.
    """
    f1 = ngram_feature.extract(corpus_instance, 'REVIEW', vocabulary)
    f2 = pos_feature.extract(corpus_instance, pos_vocabulary)
    f3 = ngram_feature.extract(corpus_instance, 'SURFACE_PATTERNS', surface_vocabulary)
    f4 = sent_rating_feature.extract(corpus_instance)
    f5 = punctuation_feature.extract(corpus_instance)
    f6 = contrast_feature.extract(corpus_instance)

    return np.concatenate((f1, f2, f3, f4, f5, f6))


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


if __name__ == '__main__':
    start_time = time.time()

    corpus = corpus.read_corpus("corpus_shuffled.csv")
    extended_corpus = surface_patterns.extract_surface_patterns(corpus, 1000)

    # split data set 80:20
    train_set = extended_corpus[:1000]
    test_set = extended_corpus[1000:]

    train_inputs, train_labels = [], []
    test_inputs, test_labels = [], []

    re_extract = True # change to False if features are unchanged since previous run

    if re_extract == True:

        # inputs (x)
        train_inputs, test_inputs = extract_features(train_set, test_set)

        # labels (y)
        train_labels = np.array([int(el['LABEL']) for el in train_set])  # 1000 labels
        test_labels = np.array([int(el['LABEL']) for el in test_set])  # 254 labels

        # save to pickle
        #pickle.dump([train_inputs, train_labels, test_inputs, test_labels], open("vectors.pickle", "wb"))

    else:
        # load from pickle
        v = pickle.load(open("vectors.pickle", "rb"))
        train_inputs, train_labels = v[0], v[1]
        test_inputs, test_labels = v[2], v[3]


    # Machine Learning

    # init
    svm_clf = svm.SVC() # best: C=0.1, gamma=0.001, kernel='linear'
    tree_clf = tree.DecisionTreeClassifier()
    nb_clf = naive_bayes.MultinomialNB()
    lr_clf = linear_model.LogisticRegression()

    # validation
    #validate_multiple([svm_clf, tree_clf, nb_clf, lr_clf], train_inputs, train_labels)
    #print("---> Duration CV: {} sec.".format(int(time.time()-start_time)))

    # tuning
    svm_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
                      'gamma' : [0.001, 0.01, 0.1, 1],
                      'kernel' : ['linear', 'rbf', 'poly']}

    tree_param_grid = {'criterion' : ['gini', 'entropy'],
                       'max_depth': [9, 6, 3, None],
                       'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 500],
                       'min_samples_leaf': randint(1, 9)}

    nb_param_grid = {'alpha' : [0, 0.5, 1.0]}

    lr_param_grid = {'penalty' : ['l1', 'l2'],
                     'C' : [0.001, 0.01, 0.1, 1, 10]}

    get_best_params(svm_clf, svm_param_grid, train_inputs, train_labels)
    get_best_params(tree_clf, tree_param_grid, train_inputs, train_labels)
    get_best_params(nb_clf, nb_param_grid, train_inputs, train_labels)
    get_best_params(lr_clf, lr_param_grid, train_inputs, train_labels)

    print("---> Duration param search: {} sec.".format(int(time.time()-start_time)))

    # training
    #train_multiple([svm_clf, tree_clf, nb_clf, lr_clf], train_inputs, train_labels)
    #print("---> Duration Training: {} sec.\n".format(int(time.time()-start_time)))

    # testing
    # print("\nSVM: Score on test Data:")
    # print(svm_clf.score(test_inputs, test_labels))
    # predictions = svm_classifier.predict(train_inputs)
