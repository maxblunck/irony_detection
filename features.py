import sent_rating_feature
import ngram_feature
import pos_feature
import punctuation_feature
import contrast_feature
import numpy as np
import config


def create_vector(corpus_instance, uni_gram_vocab=None, pos_vocabulary=None, surface_vocabulary=None, lemma_vocab=None):
    """
    Calls all feature extraction programms and combines
    resulting arrays to a single input vector (for a
    single corpus instance)
    Example for corpus instance: OrderedDict([('LABEL', '0'), ('STARS', '5.0'), etc.
    """

    # functions and their seperate arguments are stored in dict and only called when needed
    # key : (func, [args]) 
    f_map = {'f1' : (ngram_feature.extract, [corpus_instance, 'REVIEW', uni_gram_vocab]),
             'f2' : (pos_feature.extract, [corpus_instance, pos_vocabulary]), 
             'f3' : (ngram_feature.extract, [corpus_instance, 'SURFACE_PATTERNS', surface_vocabulary]), 
             'f4' : (sent_rating_feature.extract, [corpus_instance]), 
             'f5' : (punctuation_feature.extract, [corpus_instance]), 
             'f6' : (contrast_feature.extract, [corpus_instance]),
             'f7' : (extract_star_rating, [corpus_instance]),
             'f8' : (ngram_feature.extract, [corpus_instance, 'LEMMAS', lemma_vocab])
             }

    fn, args = f_map[config.feature_selection[0]]
    vector = fn(*args)
    
    if len(config.feature_selection) > 1:

        for i in range(1, len(config.feature_selection)):
            fn, args = f_map[config.feature_selection[i]]
            vector = np.append(vector, fn(*args))

    return vector


def extract_features(train_set, test_set):
    
    # vocabularies
    n_gram_vocab = None
    pos_bigram_vocab = None
    sp_n_gram_vocab = None
    lemma_n_gram_vocab = None

    print("--------Feature Extraction-------")

    if 'f1' in config.feature_selection:
        n_gram_vocab = ngram_feature.get_vocabulary(train_set, 'REVIEW', config.n_range_words) 
    if 'f2' in config.feature_selection:
        pos_bigram_vocab = pos_feature.get_pos_vocabulary(train_set)
    if 'f3' in config.feature_selection:
        sp_n_gram_vocab = ngram_feature.get_vocabulary(train_set, 'SURFACE_PATTERNS', config.n_range_surface_patterns)
    if 'f8' in config.feature_selection:
        lemma_n_gram_vocab = ngram_feature.get_vocabulary(train_set, 'LEMMAS', config.n_range_lemmas)

    # inputs:
    train_inputs = [create_vector(el, n_gram_vocab, pos_bigram_vocab, sp_n_gram_vocab, lemma_n_gram_vocab) #, bi_gram_vocab, tri_gram_vocab
                    for el in train_set]  # 1000 vectors
    test_inputs = [create_vector(el, n_gram_vocab, pos_bigram_vocab, sp_n_gram_vocab, lemma_n_gram_vocab) #, bi_gram_vocab, tri_gram_vocab
                   for el in test_set]  # 254 vectors

    # print stats
    print("Total features per train sample:  {}".format(len(train_inputs[0])))
    print("Number of train samples:          {}".format(len(train_inputs)))

    return train_inputs, test_inputs

def extract_star_rating(corpus_instance):
    return np.array([float(corpus_instance['STARS'])])




