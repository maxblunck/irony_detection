from collections import Counter
from corpus import read_corpus
from ngram_feature import get_vocabulary, extract


def extract_surface_patterns(corpus, frequency_per_million):
	corpus = read_corpus(corpus)
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


if __name__ == '__main__':

	extended_corpus = extract_surface_patterns("corpus.csv", 1000)
	
	vocabulary = get_vocabulary(extended_corpus, "SURFACE_PATTERNS", 1)

	for i in extended_corpus:
		vector = extract(i, "SURFACE_PATTERNS", vocabulary)	
		for el in vector:
			if el == 1:
				print(el)