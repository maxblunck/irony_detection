import os, os.path
import re
import csv
from nltk.tokenize import word_tokenize

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
			corpus.append(row)

	return corpus


def convert_corpus(corpus_path, out):
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

	with open(out, 'w') as csvfile:

		fieldnames = ['LABEL', 'FILENAME', 'STARS', 'TITLE', 'DATE', 'AUTHOR', 'PRODUCT', 'REVIEW', 'TOKENS']
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

			for tag in fieldnames[2:-1]:
				data[tag] = get_tag_content(tag, s)

			# tokenization
			tokens = word_tokenize(data['REVIEW'])
			data["TOKENS"] = tokens

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


if __name__ == '__main__':
	
	#corpus_path = "../corpus/SarcasmAmazonReviewsCorpus"
	#convert_corpus(corpus_path, "corpus.csv")

	#corpus = read_corpus("corpus.csv")
	#print("Corpus size: "+str(len(corpus)))





