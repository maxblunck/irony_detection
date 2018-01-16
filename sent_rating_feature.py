import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


def extract(corpus_instance):
	"""
	Extracts single "contrast" feature from a single corpus instance.
	Returns numpy array of size 1.
	"""
	review = corpus_instance["REVIEW"]
	stars = float(corpus_instance["STARS"])

    #sent = get_sent_vader(review)
	sent = get_sent_textblob(review)
	
	if (sent <= 0.0 and stars > 3.0) or (sent > 0.0 and stars < 3.0):
		return np.array([1])
	else:
		return np.array([0])


def get_sent_vader(string):
    analyser = SentimentIntensityAnalyzer()
    sent = analyser.polarity_scores(string)
    return sent['compound']


def get_sent_textblob(string):
    blob = TextBlob(string)
    return blob.sentiment.polarity


def confusion_matrix(true_labels, predicted_labels):
    matrix = np.zeros(shape=(2, 2))

    for true, pred in zip(true_labels, predicted_labels):
        matrix[true][pred] += 1

    return matrix
