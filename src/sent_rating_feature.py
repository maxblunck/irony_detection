from feature import Feature
import numpy as np
from textblob import TextBlob

class SentRatingFeature(Feature):
	"""
	Class representing feature f4

	extract-method returns a feature-vector with one value indicating
	if there is a contrast between the star rating and the sentiment
	of the review, or not

	"""

	def extract(self, corpus_instance):
		"""
		Extracts single "contrast" feature from a single corpus instance.
		Returns numpy array of size 1.
		"""
		review = corpus_instance["REVIEW"]
		stars = float(corpus_instance["STARS"])

	    #sent = self.__get_sent_vader(review)
		sent = self.__get_sent_textblob(review)
		
		if (sent <= 0.0 and stars > 3.0) or (sent > 0.0 and stars < 3.0):
			return np.array([1])
		else:
			return np.array([0])

	
	# def __get_sent_vader(self, string):
	#     analyser = SentimentIntensityAnalyzer()
	#     sent = analyser.polarity_scores(string)
	#     return sent['compound']


	def __get_sent_textblob(self, string):
	    blob = TextBlob(string)
	    return blob.sentiment.polarity


	def get_feature_names(self):
		return ['sent/rating-contrast']