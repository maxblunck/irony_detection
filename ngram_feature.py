from sklearn.feature_extraction.text import CountVectorizer


def extract(corpus_instance, corpus_dict_key, vocabulary):
	"""
	Extracts n-gram features from a single corpus instance.
	n depends on vocabulary, which needs to be extracted using get_vocabulary.
	Returns numpy array of size of vocabulary
	"""
	n = len(list(vocabulary.keys())[0].split())
	vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=(n, n))
	
	vector = vectorizer.transform([corpus_instance[corpus_dict_key]]) # takes a list

	return vector.toarray()[0]


def get_vocabulary(corpus, corpus_dict_key, n_range):
	"""
	Creates vocabulary based on given corpus.
	"""
	all_reviews = []
	for line in corpus:
		all_reviews.append(line[corpus_dict_key])

	vectorizer = CountVectorizer(ngram_range=n_range)
	vectorizer.fit(all_reviews)

	return vectorizer.vocabulary_
