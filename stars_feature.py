from feature import Feature
import numpy as np

class StarsFeature(Feature):
	"""
	Class representing feature f7

	extract-method returns a feature-vector with one value
	holding the number of stars of a review

	"""

	def extract(self, corpus_instance):
		return np.array([float(corpus_instance['STARS'])])


	def get_feature_names(self):
		return ['number_of_stars']