import corpus
import sent_rating_feature
import ngram_feature
import pos_feature
import punctuation_feature
import surface_patterns
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn.model_selection import cross_val_score


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

    return np.concatenate((f1, f2, f3, f4, f5))


def train_multiple(classifiers, train_input, train_labels):
    for classifier in classifiers:
        classifier.fit(train_input, train_labels)


def score_multiple(classifiers, train_input, train_labels):
    scores = []
    for classifier in classifiers:
        accuracy = cross_val_score(classifier, train_inputs, train_labels, cv=5, scoring='accuracy').mean()
        f1 = cross_val_score(classifier, train_inputs, train_labels, cv=5, scoring='f1').mean()
        scores.append(accuracy, f1)
    return scores


if __name__ == '__main__':

    corpus = corpus.read_corpus("corpus_shuffled.csv")
    extended_corpus = surface_patterns.extract_surface_patterns(corpus, 1000)

    # split data set 80:20
    train_set = extended_corpus[:1000]
    test_set = extended_corpus[1000:]

    # vocabularies
    unigram_vocab = ngram_feature.get_vocabulary(train_set, 'REVIEW', 1)
    bigram_vocab = ngram_feature.get_vocabulary(train_set, 'REVIEW', 2)
    pos_bigram_vocab = pos_feature.get_pos_vocabulary(train_set)
    surface_bigram_vocab = ngram_feature.get_vocabulary(train_set, 'SURFACE_PATTERNS', 2)

    # inputs:
    train_inputs = [create_vector(el, bigram_vocab, pos_bigram_vocab, surface_bigram_vocab)
                    for el in train_set]  # 1000 vectors
    #test_inputs = [create_vector(el, bigram_vocab, pos_bigram_vocab, surface_bigram_vocab)
    #               for el in test_set]  # 254 vectors

    # labels
    train_labels = np.array([int(el['LABEL']) for el in train_set])  # 1000 labels
    test_labels = np.array([int(el['LABEL']) for el in test_set])  # 254 labels

    print("Number of train samples:             {}".format(len(train_inputs)))
    print("Number of features per train sample: {}".format(len(train_inputs[0])))
    print("Unigram vocab size:                  {}".format(len(unigram_vocab)))
    print("Bigram vocab size:                   {}".format(len(bigram_vocab)))
    print("POS-Bigram vocab size:               {}".format(len(pos_bigram_vocab)))
    print("Surface Patterns-Bigram vocab size:  {}".format(len(surface_bigram_vocab)))

    # TODO Pickle/outsource

    # ML

    # init
    svm_clf = svm.SVC(C=200.0, kernel='linear') # large C: smaller-margin hyperplane
    tree_clf = tree.DecisionTreeClassifier()
    nb_clf = naive_bayes.MultinomialNB()
    lr_clf = linear_model.LogisticRegression()

    # training
    train_multiple([svm_clf, tree_clf, nb_clf, lr_clf], train_inputs, train_labels)

    # validation
    svm_acc = cross_val_score(svm_clf, train_inputs, train_labels, cv=5, scoring='accuracy').mean()
    tree_acc = cross_val_score(tree_clf, train_inputs, train_labels, cv=5, scoring='accuracy').mean()
    nb_acc = cross_val_score(nb_clf, train_inputs, train_labels, cv=5, scoring='accuracy').mean()
    lr_acc = cross_val_score(lr_clf, train_inputs, train_labels, cv=5, scoring='accuracy').mean()

    svm_f1 = cross_val_score(svm_clf, train_inputs, train_labels, cv=5, scoring='f1').mean()
    tree_f1 = cross_val_score(tree_clf, train_inputs, train_labels, cv=5, scoring='f1').mean()
    nb_f1 = cross_val_score(nb_clf, train_inputs, train_labels, cv=5, scoring='f1').mean()
    lr_f1 = cross_val_score(lr_clf, train_inputs, train_labels, cv=5, scoring='f1').mean()

    print("\n--Cross Validation Scores-- ")
    print("\nSVM: Accuracy: {}, F1-Score: {}".format(svm_acc, svm_f1))
    print("\nTree: Accuracy: {}, F1-Score: {}".format(tree_acc, tree_f1))
    print("\nN. Bayes: Accuracy: {}, F1-Score: {}".format(nb_acc, nb_f1))
    print("\nLog. Regression: Accuracy: {}, F1-Score: {}".format(lr_acc, lr_f1))

    # testing
    # print("\nSVM: Score on test Data:")
    # print(svm_clf.score(test_inputs, test_labels))

    # print("\nDTree: Score on test Data:")
    # print(tree_clf.score(test_inputs, test_labels))

    # predictions = svm_classifier.predict(train_inputs)
