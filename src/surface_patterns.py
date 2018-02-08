from collections import Counter
from ngram_feature import NgramFeature
import config


class SurfacePatternFeature(NgramFeature):
	"""
	Class representing feature f3

	extract-method returns a feature-vector of length of its vocabulary 
	containing surface-pattern-n-gram counts

	"""

	corpus_key = 'SURFACE_PATTERNS'
	n_range = config.n_range_surface_patterns
	name = "Surface Pattern"


# below are functions for creating surface patterns when loading corpus
def extract_surface_patterns(corpus, frequency_per_million):
	tokens = get_tokens_lower(corpus)
	frequencies = frequency_breakdown(tokens)
	threshold = frequency_threshold(tokens, frequency_per_million)
	hfws = get_high_frequency_words(frequencies, threshold)
	extended_corpus = substitute_content_words(corpus, hfws)
	return extended_corpus


def get_tokens_lower(corpus):
	tokens = []
	for instance in corpus:
		for token in instance["TOKENS"]:
			tokens.append(token.lower())
	return tokens


def frequency_breakdown(tokens):	
	frequencies = Counter(tokens)
	return frequencies


def frequency_threshold(tokens, frequency_per_million):
	length = len(tokens)
	ratio = length/1000000
	threshold = int(ratio*frequency_per_million)
	return threshold


def get_high_frequency_words(frequencies, threshold):
	hfws = []
	for key, value in frequencies.items():
		if value >= threshold:
			hfws.append(key)
	return sorted(hfws)


def substitute_content_words(corpus, hfws):
	extended_corpus = corpus
	for instance in extended_corpus:
		sub_tokens = []
		for token in instance["TOKENS"]:
			if not token.lower() in hfws:
				token = "CW"
			sub_tokens.append(token)
		instance["SURFACE_PATTERNS"] = " ".join(sub_tokens)
	return extended_corpus
