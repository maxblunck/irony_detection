import os, os.path
import re
import csv
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from random import shuffle


def read_corpus(csv_corpus_path):
	"""
	Reads a csv-file and returns a list of dicts. 
	Each dict represents one corpus file.
	Keys: ['LABEL', 'FILENAME', 'STARS', 'TITLE', 'DATE', 'AUTHOR', 'PRODUCT', 'REVIEW', 'TOKENS']
	"""
	corpus = []

	with open(csv_corpus_path) as csvfile:
		reader = csv.DictReader(csvfile)

		for row in reader:
			data = row

			# tokenization
			data["TOKENS"] = word_tokenize(row['REVIEW'])

			# pos-tagging
			data["POS"] = nltk.pos_tag(data["TOKENS"])

			# lemmatizing
			data["LEMMAS"] = get_lemmas(data["POS"])

			corpus.append(data)

	return corpus


def convert_corpus(corpus_path, out, shuffle_corpus=False):
	"""
	Takes root path of raw Filatrova corpus and converts it into a single csv file.
	
	"""
	corpus_files = []

	for root, dirs, files in os.walk(corpus_path):
	    
	    for name in files:

	        if name.endswith((".txt")):
	        	parent = root.split("/")[-1]

	        	if parent == "Regular" or parent == "Ironic":
	        		corpus_files.append(os.path.join(root, name))

	if shuffle_corpus == True:
		shuffle(corpus_files)

	with open(out, 'w') as csvfile:

		fieldnames = ['LABEL', 'FILENAME', 'STARS', 'TITLE', 'DATE', 'AUTHOR', 'PRODUCT', 'REVIEW']
		writer = csv.DictWriter(csvfile, fieldnames)

		writer.writeheader()

		for file_path in corpus_files:

			file = open(file_path, encoding="ISO-8859-1")
			s = file.read()
			data = {}

			label = file_path.split("/")[-2]

			if label == "Ironic":
				data[fieldnames[0]] = 1
			elif label == "Regular":
				data[fieldnames[0]] = 0
			else:
				raise ValueError("Label Error!")

			data[fieldnames[1]] = file_path.split("/")[-1]

			for tag in fieldnames[2:]:
				data[tag] = get_tag_content(tag, s)

			writer.writerow(data)

	print("Corpus written to: "+out)


def get_tag_content(tag, text):
	"""
	Helper for getting content between two xml-like tags

	"""
	pattern = r'<' + re.escape(tag) + r'>((?:\n|.)*?)</' + re.escape(tag) + r'>'
	match = re.findall(pattern, text)

	if len(match) != 1:
		raise ValueError("Matching error!")

	return match[0].strip()


def get_lemmas(instance_pos_tags):
	lemmatizer = WordNetLemmatizer()
	pos_map = {"VB" : "v", "NN" : "n", "JJ" : "a"}
	lemmas = []

	for pair in instance_pos_tags:
			token = pair[0]
			pos_tag = pair[1][0:2]

			simple_pos = "n"

			if pos_tag in pos_map.keys():
				simple_pos = pos_map[pos_tag]

			lemma = lemmatizer.lemmatize(token, pos=simple_pos)
			lemmas.append(lemma)

	return lemmas




if __name__ == '__main__':
	"""
	corpus_path = "../corpus/SarcasmAmazonReviewsCorpus"
	convert_corpus(corpus_path, "corpus.csv")
	convert_corpus(corpus_path, "corpus_shuffled.csv", shuffle_corpus=True)

	corpus = read_corpus("corpus.csv")
	print("Corpus size: "+str(len(corpus)))
	print(corpus[0].keys())
	"""



