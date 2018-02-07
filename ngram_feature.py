from feature import Feature
from sklearn.feature_extraction.text import CountVectorizer
import config

class NgramFeature(Feature):
	"""
	Class representing feature f1

	extract-method returns a feature-vector of length of its vocabulary 
	containing n-gram counts
	
	"""

	name = "Bag-of-ngram"
	corpus_key = 'REVIEW'
	n_range = config.n_range_words
	vocabulary = None
	vectorizer = None
	

	# def __init__(self, lemmatize=False):
	# 	#TODO if lemmatize == True


	def extract(self, corpus_instance):
		"""
		Extracts n-gram features from a single corpus instance.
		Returns numpy array of size of vocabulary
		"""
		vector = self.vectorizer.transform([corpus_instance[self.corpus_key]]) # takes a list
		return vector.toarray()[0]

		
	def load_vocabulary(self, corpus):
		"""
		Creates vocabulary based on given corpus (Only train-data!).
		"""
		all_reviews = []

		for line in corpus:
			all_reviews.append(line[self.corpus_key])

		vectorizer = CountVectorizer(ngram_range=self.n_range)
		vectorizer.fit(all_reviews)

		self.vectorizer = vectorizer
		self.vocabulary = vectorizer.vocabulary_

		if config.print_stats == True:
			print("{} Vocab size (n={},{}):\t{}".format(self.name ,self.n_range[0], self.n_range[1], len(self.vocabulary)))


	def get_feature_names(self):
		'''
		Turn vocabulary dict. into list, where indices are equal to indices-keys of dict
		'''
		return sorted(self.vocabulary, key=self.vocabulary.get)
