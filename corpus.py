import os, os.path
import re
import csv
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import config
import surface_patterns
from random import shuffle
import io
import sys


def read_corpus(csv_corpus_path):
	"""
	Reads a csv-file containing the corpus and returns a list of dicts. 
	Each dict represents one corpus file.
	Keys: ['LABEL', 'FILENAME', 'STARS', 'TITLE', 'DATE', 'AUTHOR', 'PRODUCT', 'REVIEW', 'TOKENS']
	"""
	corpus = []

	with io.open(csv_corpus_path, encoding="ISO-8859-1") as csvfile:
		reader = csv.DictReader(csvfile)

		# augment corpus with tokens, pos-tags & lemmas
		for row in reader:
			data = row

			data["TOKENS"] = word_tokenize(row['REVIEW'])

			data["POS"] = nltk.pos_tag(data["TOKENS"])

			data["LEMMAS"] = get_lemmas(data["POS"])

			corpus.append(data)

	# add surface patterns to corpus data
	extended_corpus = surface_patterns.extract_surface_patterns(corpus, config.sp_threshold)

	return extended_corpus


def convert_corpus(corpus_path, out, shuffle_corpus=False):
	"""
	Takes root path of raw Filatrova corpus and converts it into a single csv file.
	
	"""
	corpus_files = []

	# gather all text files of directory
	for root, dirs, files in os.walk(corpus_path):
	    
	    for name in files:

	        if name.endswith((".txt")):
	        	parent = root.split("/")[-1]

	        	if parent == "Regular" or parent == "Ironic":
	        		corpus_files.append(os.path.join(root, name))

	# shuffle files
	if shuffle_corpus == True:
		shuffle(corpus_files)

	with open(out, 'w') as csvfile:

		fieldnames = ['LABEL', 'FILENAME', 'STARS', 'TITLE', 'DATE', 'AUTHOR', 'PRODUCT', 'REVIEW']
		writer = csv.DictWriter(csvfile, fieldnames)

		writer.writeheader()

		for file_path in corpus_files:
			try:
				file = open(file_path)

				content = None

				try:
					content = file.read()
				except UnicodeDecodeError:
					# handle encoding problems
					file = open(file_path, encoding="ISO-8859-1")
					content_iso = file.read()
					content = content_iso.encode('utf-8').decode('utf-8')

				data = {}

				label = file_path.split("/")[-2]

				if label == "Ironic":
					data[fieldnames[0]] = 1
				elif label == "Regular":
					data[fieldnames[0]] = 0
				else:
					raise ValueError("Label Error!")

				data[fieldnames[1]] = file_path.split("/")[-1]

				# get all the content between xml-like tags
				for tag in fieldnames[2:]:
						data[tag] = get_tag_content(tag, content)

				writer.writerow(data)

			except ValueError:
				# just skip files that do not match the regular file structure
				continue	

	print("Corpus as single file written to: "+out)


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
	'''
	Helper for lemmatizing tokens of corpus
	'''
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
	try:
		corpus_source_path = sys.argv[1]
		csv_out_path = sys.argv[2]

		convert_corpus(corpus_source_path, csv_out_path, shuffle_corpus=True)

		corpus = read_corpus(csv_out_path)
		print("Corpus size: "+str(len(corpus)))

	except IndexError:
		print("""Please provide following arguments:
 $ python3 corpus.py [corpus_source] [csv_out] 
 [corpus_source] - path to root folder of Filatrova's Review Corpus
 [csv_out] - path and file name of csv-file to save corpus to""")



