import corpus
import nltk
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_phrase(i, n, tokens_only, tags_only):
	#fourgram: n=4 
	try:
		pos_sent_phrase = tokens_only[i]
		neg_situation_phrase = " ".join(tokens_only[(i+1):(i+n)])

		try:
			if tags_only[i-1] == 'R':
				pos_sent_phrase = tokens_only[i-1] +" "+ pos_sent_phrase
		except IndexError:
			return (pos_sent_phrase, neg_situation_phrase)

		return (pos_sent_phrase, neg_situation_phrase)

	except IndexError:
		pass

def extract(corpus_instance):
	tokens = corpus_instance['TOKENS']
	tagged = nltk.pos_tag(tokens)

	tags_only = [y[0] for (x,y) in tagged]
	tokens_only = [x for (x,y) in tagged]

	# pos sentiment phrases
	verb_phrase_list = ["V"]

	# only situation pos-tag combos like the following should be matched
	uni_pos_list = ["V"]
	bi_pos_list = ["VV", "VR", "RV", "TV", "VN", "VN", "VN", "VP", "VJ"]
	tri_pos_list = ["VVV", "VVR", "VRV", "VVR", "VRR", "RVV", "VNR", "VIN", "VTV", "VIP"] 
	excl_N_tri_pos_list = ["VVN", "VNN", "VJN", "VDN", "RVN"] # -JN = next tag is not J/N
	excl_JN_tri_pos_list = ["VRJ", "VVJ", "VRJ", "RVJ"]

	# generate possible pos-tag comintations
	phrase_patterns = []
	excl_N_phrase_patterns = []
	excl_JN_phrase_patterns = []

	for a in verb_phrase_list:
		for b in uni_pos_list:
			phrase_patterns.append(a+b)
		for c in bi_pos_list:
			phrase_patterns.append(a+c)
		for d in tri_pos_list:
			phrase_patterns.append(a+d)
		for e in excl_N_tri_pos_list:
			excl_N_phrase_patterns.append(a+e)
		for f in excl_JN_tri_pos_list:
			excl_JN_phrase_patterns.append(a+f)
  
	contrasts = 0
	candidates = []

	# get all phrases matching the patterns
	#TODO: elim doubles
	for i in range(len(tags_only)):

		fourgram = "".join(tags_only[i:(i+4)])
		trigram = "".join(tags_only[i:(i+3)])
		bigram = "".join(tags_only[i:(i+2)])

		if fourgram in phrase_patterns:
			candidates.append(get_phrase(i, 4, tokens_only, tags_only))

		elif fourgram in excl_N_phrase_patterns:
			try:
				if tokens_only[i+4] != 'N':
					candidates.append(get_phrase(i, 4, tokens_only, tags_only))
			except IndexError:
				candidates.append(get_phrase(i, 4, tokens_only, tags_only))

		elif fourgram in excl_JN_phrase_patterns:
			try:
				if tokens_only[i+4] != 'N' and tokens_only[i+4] != 'J':
					candidates.append(get_phrase(i, 4, tokens_only, tags_only))
			except IndexError:
				candidates.append(get_phrase(i, 4, tokens_only, tags_only))
				
		elif trigram in phrase_patterns:
			candidates.append(get_phrase(i, 3, tokens_only, tags_only))

		elif bigram in phrase_patterns:
			candidates.append(get_phrase(i, 2, tokens_only, tags_only))


	# determine sentiment of extracted phrased
	if candidates != []:
		for phrase in candidates:
			verb = phrase[0]
			situation = phrase[1]

			analyser = SentimentIntensityAnalyzer()
			sent_verb = analyser.polarity_scores(verb)['compound']
			sent_situation = analyser.polarity_scores(situation)['compound']

			if (sent_verb > 0.0 and sent_situation < 0.0) or (sent_verb < 0.0 and sent_situation > 0.0):
				#print("phrase: {} {} sent verb: {}  sent situation: {}".format(verb, situation, sent_verb, sent_situation))
				contrasts += 1

	return np.array([contrasts])


# if __name__ == '__main__':
# 	corpus = corpus.read_corpus("corpus_shuffled.csv")

# 	for instance in corpus:
# 		extract(instance)

