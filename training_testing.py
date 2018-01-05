import corpus
from random import shuffle
import sent_rating_feature
import ngram_feature
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import postagger


def create_vector(corpus_instance, vocabulary=None, pos_vocabulary=None):
    """
    Calls all feature extraction programms and combines
    resulting arrays to a single input vector (for a 
    single corpus instance)
    """
    f1 = ngram_feature.extract(corpus_instance, vocabulary)
    # f2 = postagger.to_bigram_vector(corpus_instance, pos_vocabulary)
    f4 = sent_rating_feature.extract(corpus_instance)

    return np.concatenate((f1,f4))


if __name__ == '__main__':

    corpus = corpus.read_corpus("corpus.csv")

    # shuffle & split data set 80:20
    shuffle(corpus)
    train_set = corpus[:1000]
    test_set = corpus[1000:]

    # vocabularies
    unigram_vocab = ngram_feature.get_vocabulary(train_set, 1)
    bigram_vocab = ngram_feature.get_vocabulary(train_set, 2)
    
    # pos_bags
    # bigram_pos_vocab = postagger.get_pos_vocabulary(train_set) (entspricht corpus in postagger.py)
    

    # inputs:
    train_inputs = [create_vector(el, unigram_vocab)
                    for el in train_set]  # 1000 vectors
    test_inputs = [create_vector(el, unigram_vocab)
                   for el in test_set]  # 254 vectors

    # labels
    train_labels = np.array([int(el['LABEL']) for el in train_set])  # 1000 labels
    test_labels = np.array([int(el['LABEL']) for el in test_set])  # 254 labels

    print("Number of train samples:             {}".format(len(train_inputs)))
    print("Number of features per train sample: {}".format(len(train_inputs[0])))
    print("Unigram vocab size:                  {}".format(len(unigram_vocab)))
    print("Bigram vocab size:                   {}".format(len(bigram_vocab)))

    # training

    # SVM
    svm_classifier = svm.SVC()
    svm_classifier.fit(train_inputs, train_labels)
    predictions = svm_classifier.predict(test_inputs)
    print(svm_classifier.score(test_inputs, test_labels))
    print("Predictions: \n {}".format(predictions))
    print("Targets:     \n {}".format(test_labels))

    # Trees
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(train_inputs, train_labels)
    predictions = tree_clf.predict(test_inputs)
    print(tree_clf.score(test_inputs, test_labels))
    print("Predictions: \n {}".format(predictions))
    print("Targets:     \n {}".format(test_labels))


