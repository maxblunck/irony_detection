import itertools as it
import config
import csv
import numpy as np

"""
Collection of helpers and utilities used throughout the app

"""

def get_all_feature_names(features):
    """
    Takes a list of Feature objects.
    Returns a concatenated list of all feature names/descriptions.
    """
    names = features[0].get_feature_names()

    if len(features) > 1:
        for i in range(1, len(features)):
            current_names = features[i].get_feature_names()
            names += current_names

    return names


def print_trees_top_features(tree_clf, feature_names):
    """
    Takes a Dec. Tree classifier and list of feature names.
    Prints most important features used by the Dec. Tree.
    """
    print("Best tree features")
    best = tree_clf.feature_importances_
    for i in range(len(best)):
        if best[i] != 0:
            print("Feature: {} Value: {}".format(feature_names[i], best[i]))


def get_feature_combos():
    """
    Generates a list of all combinations
    of the feautures provided in the config
    """
    f_combinations = []

    # add all possible feature-combos to array
    for i in range(1, len(config.feature_selection)+1):
        combinations = it.combinations(config.feature_selection, i)
        for combo in combinations:
            f_combinations.append(list(combo))

    return f_combinations


def plot_coefficients(classifier, feature_names, top_features=20):
    """
    Plots most important weights of SVM/Log Reg. classifier
    Code mostly copied from "Visualising Top Features in Linear SVM" 
    by Aneesha Bakharia on medium.com
    """
    import matplotlib.pyplot as plt # leave here, as lib cant load on ella

    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    
    print("\nRunning plotter.. Close open windows to continue.")
    plt.show()


def get_misclassifications(classifier, test_set, test_inputs, test_labels):
    """
    Write missclassified test instances to file for a given classifier
    """
    try:
        predictions = classifier.predict(test_inputs)
        out = open(config.misclassification_out, 'w')

        for i in range(len(predictions)):
            if predictions[i] != test_labels[i]:

                out.write("Prediction: {}\n".format(predictions[i]))
                out.write("True      : {}\n".format(test_set[i]["LABEL"]))
                out.write("File: {}\n".format(test_set[i]["FILENAME"]))
                out.write("Stars: {}".format(test_set[i]["STARS"]+"\n"))
                out.write(test_set[i]["TITLE"]+"\n")
                out.write(test_set[i]["REVIEW"]+"\n")
                out.write("\n")
                out.write("\n")

        print("\nMisclassifications saved to: {}".format(config.misclassification_out))
    except ValueError:
        print("Missclassification report unavailable. Probably no test data given.")


def write_resultlog_headers():
    test_csv_file = open(config.test_result_out, "w")
    valid_csv_file = open(config.validation_result_out, "w")

    fields = ["classifier", "features", "accuracy", "precision", "recall", "f1-score"]

    writer_test = csv.DictWriter(test_csv_file, fields)
    writer_valid = csv.DictWriter(valid_csv_file, fields)

    writer_test.writeheader()
    writer_valid.writeheader()

    test_csv_file.close()
    valid_csv_file.close()


def confusion_matrix(true_labels, predicted_labels):
    matrix = np.zeros(shape=(2, 2))

    for true, pred in zip(true_labels, predicted_labels):
        matrix[true][pred] += 1

    return matrix
