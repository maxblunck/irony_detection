import corpus
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def extract(corpus_instance):
	tokens = corpus_instance['TOKENS']
	tagged = nltk.pos_tag(tokens)

	# only pos-tag combos like the following should be matched
	uni_pos_list = ["VB"]
	bi_pos_list = ["VBVB", "VBRB", "RBVB", "TOVB", "VBNN", "VBNNP", "VBNNS", "VBPRP", "VBPRP$", "VBJJ", "VBJJS"]
	#tri_pos_list = []

	candidates = []

	# go through all tags and find phrases: VB + tag-combo of list above
	for i in range(len(tagged)):
		if i+4 <= len(tagged):

			# phrase should begin with verb
			if tagged[i][1] == 'VB':

				uni_pos = tagged[i+1][1]
				bi_pos = tagged[i+1][1] + tagged[i+2][1]
				#tri_pos = tagged[i+1][1] + tagged[i+2][1] + tagged[i+3][1]

				#if tri_pos in tri_pos_list:

				if bi_pos in bi_pos_list:

					phrase = tagged[i:(i+3)]
					candidates.append(phrase)

				elif uni_pos in uni_pos_list:

					phrase = tagged[i:(i+2)]
					candidates.append(phrase)

	# determine sentiment of extracted phrased
	if candidates != []:
		for phrase in candidates:
			verb = phrase[0][0]
			situation = ""
			for word in phrase[1:len(phrase)]:
				situation += word[0] + " "

			analyser = SentimentIntensityAnalyzer()
			sent_verb = analyser.polarity_scores(verb)['compound']
			sent_situation = analyser.polarity_scores(situation)['compound']

			print("phrase: {} {} sent verb: {}  sent situation: {}".format(verb, situation, sent_verb, sent_situation))



if __name__ == '__main__':
	corpus = corpus.read_corpus("corpus_shuffled.csv")[:100]

	for instance in corpus:
		extract(instance)