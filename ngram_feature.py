from sklearn.feature_extraction.text import CountVectorizer


def extract(corpus_instance, vocabulary):
	"""
	Extracts n-gram features from a single corpus instance.
	n depends on vocabulary, which needs to be extracted using get_vocabulary.
	Returns numpy array of size of vocabulary
	"""
	vectorizer = CountVectorizer(vocabulary=vocabulary)
	vector = vectorizer.transform([corpus_instance['REVIEW']]) # takes a list

	return vector.toarray()[0]


def get_vocabulary(corpus, n):
	"""
	Creates vocabulary based on given corpus.
	"""
	all_reviews = []
	for line in corpus:
		all_reviews.append(line['REVIEW'])

	vectorizer = CountVectorizer(ngram_range=(n, n))
	vectorizer.fit(all_reviews)

	return vectorizer.vocabulary_
