from feature import Feature
import nltk
import numpy as np
from textblob import TextBlob


class ContrastFeature(Feature):
	"""
	Class representing feature f6, based on Riloff et al. (2013)

	extract-method returns a feature-vector of length 1 containing the number of 
	contrasts found in a review

	"""

	def get_feature_names(self):
		return ['riloff-contrast']


	def extract(self, corpus_instance):
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

		# generate possible pos-tag combintations
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
				candidates.append(self.__get_phrase(i, 4, tokens_only, tags_only))

			elif fourgram in excl_N_phrase_patterns:
				try:
					if tokens_only[i+4] != 'N':
						candidates.append(self.__get_phrase(i, 4, tokens_only, tags_only))
				except IndexError:
					candidates.append(self.__get_phrase(i, 4, tokens_only, tags_only))

			elif fourgram in excl_JN_phrase_patterns:
				try:
					if tokens_only[i+4] != 'N' and tokens_only[i+4] != 'J':
						candidates.append(self.__get_phrase(i, 4, tokens_only, tags_only))
				except IndexError:
					candidates.append(self.__get_phrase(i, 4, tokens_only, tags_only))
					
			elif trigram in phrase_patterns:
				candidates.append(self.__get_phrase(i, 3, tokens_only, tags_only))

			elif bigram in phrase_patterns:
				candidates.append(self.__get_phrase(i, 2, tokens_only, tags_only))


		# determine sentiment of extracted phrased
		if candidates != []:
			for phrase in candidates:
				verb = phrase[0]
				situation = phrase[1]

				sent_verb = TextBlob(verb).sentiment.polarity
				sent_situation = TextBlob(situation).sentiment.polarity

				# if verb and situation are in contrast to another: increase feature value by one
				if (sent_verb > 0.0 and sent_situation < 0.0) or (sent_verb < 0.0 and sent_situation > 0.0):
					#print("phrase: {} {} sent verb: {}  sent situation: {}".format(verb, situation, sent_verb, sent_situation))
					contrasts += 1

		return np.array([contrasts])


	def __get_phrase(self, i, n, tokens_only, tags_only):
		# builds phrase corresponding to the matched POS-tag-combo
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
