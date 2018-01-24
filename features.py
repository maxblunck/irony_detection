import sent_rating_feature
import ngram_feature
import pos_feature
import punctuation_feature
import contrast_feature
import numpy as np
import config


def create_vector(corpus_instance, uni_gram_vocab=None, pos_vocabulary=None, surface_vocabulary=None):
    """
    Calls all feature extraction programms and combines
    resulting arrays to a single input vector (for a
    single corpus instance)
    Example for corpus instance: OrderedDict([('LABEL', '0'), ('STARS', '5.0'), etc.
    """

    f_map = {'f1' : ngram_feature.extract(corpus_instance, 'REVIEW', uni_gram_vocab),
             'f2' : pos_feature.extract(corpus_instance, pos_vocabulary), 
             'f3' : ngram_feature.extract(corpus_instance, 'SURFACE_PATTERNS', surface_vocabulary), 
             'f4' : sent_rating_feature.extract(corpus_instance), 
             'f5' : punctuation_feature.extract(corpus_instance), 
             'f6' : contrast_feature.extract(corpus_instance)
             }

    vector = f_map[config.feature_selection[0]]
    
    if len(config.feature_selection) > 1:

        for i in range(1, len(config.feature_selection)):
            vector = np.append(vector, f_map[config.feature_selection[i]])

    return vector


def extract_features(train_set, test_set):

    # vocabularies
    n_gram_vocab = ngram_feature.get_vocabulary(train_set, 'REVIEW', config.n_range_words) 
    pos_bigram_vocab = pos_feature.get_pos_vocabulary(train_set)
    surface_bigram_vocab = ngram_feature.get_vocabulary(train_set, 'SURFACE_PATTERNS', config.n_range_surface_patterns)

    # inputs:
    print("------Feature Extraction------\n")
    train_inputs = [create_vector(el, n_gram_vocab, pos_bigram_vocab, surface_bigram_vocab) #, bi_gram_vocab, tri_gram_vocab
                    for el in train_set]  # 1000 vectors
    test_inputs = [create_vector(el, n_gram_vocab, pos_bigram_vocab, surface_bigram_vocab) #, bi_gram_vocab, tri_gram_vocab
                   for el in test_set]  # 254 vectors

    # stats
    print("Number of train samples:          {}".format(len(train_inputs)))
    print("N_gram vocab size:                {}".format(len(n_gram_vocab)))
    print("POS-Bigram vocab size:            {}".format(len(pos_bigram_vocab)))
    print("SP-Bigram vocab size:             {}".format(len(surface_bigram_vocab)))
    print("Total features per train sample:  {}".format(len(train_inputs[0])))
    #print("---> Duration Feature Extraction: {} sec.\n".format(int(time.time()-start_time)))

    return train_inputs, test_inputs



