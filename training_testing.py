import corpus
from random import shuffle
import sent_rating_feature
import ngram_feature
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import cross_val_score
import postagger


def create_vector(corpus_instance, vocabulary=None, pos_vocabulary=None):
    """
    Calls all feature extraction programms and combines
    resulting arrays to a single input vector (for a 
    single corpus instance)
    Example for corpus instance: OrderedDict([('LABEL', '0'), ('FILENAME', '36_19_RPRRQDRSHDV6J.txt'), ('STARS', '5.0'), ('TITLE', etc.
    """
    f1 = ngram_feature.extract(corpus_instance, vocabulary)
    f2 = postagger.extract(corpus_instance, pos_vocabulary)
    f4 = sent_rating_feature.extract(corpus_instance)

    print(f2)
    print(len(f2))

    return np.concatenate((f1, f2, f4))


def train_multiple(classifiers, train_input, train_labels):
    for classifier in classifiers:
        classifier.fit(train_input, train_labels)


if __name__ == '__main__':

    corpus = corpus.read_corpus("corpus_shuffled.csv")

    # split data set 80:20
    train_set = corpus[:1000]
    test_set = corpus[1000:]

    # vocabularies
    unigram_vocab = ngram_feature.get_vocabulary(train_set, 1)
    bigram_vocab = ngram_feature.get_vocabulary(train_set, 2)
    
    # pos_bags
    pos_bigram_vocab = postagger.get_pos_vocabulary(train_set)
    #print(pos_bigram_vocab) #already lookin' good
    
    # inputs:
    train_inputs = [create_vector(el, unigram_vocab, pos_bigram_vocab)
                    for el in train_set]  # 1000 vectors
    test_inputs = [create_vector(el, unigram_vocab, pos_bigram_vocab)
                   for el in test_set]  # 254 vectors

    # labels
    train_labels = np.array([int(el['LABEL']) for el in train_set])  # 1000 labels
    test_labels = np.array([int(el['LABEL']) for el in test_set])  # 254 labels

    print("Number of train samples:             {}".format(len(train_inputs)))
    print("Number of features per train sample: {}".format(len(train_inputs[0])))
    print("Unigram vocab size:                  {}".format(len(unigram_vocab)))
    print("Bigram vocab size:                   {}".format(len(bigram_vocab)))
    print("POS-Bigram vocab size:               {}".format(len(pos_bigram_vocab)))

    # TODO Pickle/outsource

    # ML

    # init
    svm_clf = svm.SVC(C=200.0) # large C: smaller-margin hyperplane
    tree_clf = tree.DecisionTreeClassifier()

    # training
    train_multiple([svm_clf, tree_clf], train_inputs, train_labels)
    
    # validation
    svm_score = cross_val_score(svm_clf, train_inputs, train_labels, cv=5).mean()#, scoring='f1')
    tree_score = cross_val_score(tree_clf, train_inputs, train_labels, cv=5).mean()#, scoring='f1')

    print("\n--Cross Validation Scores-- ")
    print("\nSVM: {}".format(svm_score))
    print("\nTree: {}".format(tree_score))

    # testing
    # print("\nSVM: Score on test Data:")
    # print(svm_clf.score(test_inputs, test_labels))

    # print("\nDTree: Score on test Data:")
    # print(tree_clf.score(test_inputs, test_labels))

    # predictions = svm_classifier.predict(train_inputs)


