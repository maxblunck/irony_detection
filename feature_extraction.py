import sent_rating_feature
import ngram_feature
import pos_feature
import punctuation_feature
import contrast_feature
import surface_patterns
import stars_feature
import numpy as np
import config


def extract_features(train_set, test_set):
    """
    Extracts feature vectors of given train/test set.
    Extraction based on selected features in config file.
    Returns lists of feature vectors and a list of feature objects
    for further use.

    """
    f_selection_map = {'f1' : ngram_feature.NgramFeature(),
                       'f2' : pos_feature.PosFeature(), 
                       'f3' : surface_patterns.SurfacePatternFeature(),
                       'f4' : sent_rating_feature.SentRatingFeature(), 
                       'f5' : punctuation_feature.PunctuationFeature(), 
                       'f6' : contrast_feature.ContrastFeature(),
                       'f7' : stars_feature.StarsFeature()
                       }

    # get all feature objects of features selected in config
    features = [f_selection_map[feat] for feat in config.feature_selection]

    # load vocabulary if needed for feature
    for feature in features:
        try:
            feature.load_vocabulary(train_set)
        except AttributeError:
            continue

    train_inputs = [create_input_vector(features, instance) for instance in train_set]
    test_inputs = [create_input_vector(features, instance) for instance in test_set]

    # print stats
    print("\nTotal features per train sample:\t{}".format(len(train_inputs[0])))
    print("Number of train samples:\t\t{}".format(len(train_inputs)))

    return train_inputs, test_inputs, features


def create_input_vector(features, corpus_instance):
    """
    Create a feature vector for a single corpus instance
    """
    vector = features[0].extract(corpus_instance)

    if len(features) > 1:
        for i in range(1, len(features)):
            current_vec = features[i].extract(corpus_instance)
            vector = np.append(vector, current_vec)

    return vector
    