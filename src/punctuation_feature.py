from feature import Feature
from nltk.tokenize import word_tokenize
import re

class PunctuationFeature(Feature):
    """
    Class representing feature f5

    extract-method returns a feature-vector of length 8, containing
    punctuation based features

    """

    def extract(self, corpus_instance):
        relevant_punctuation = ['!', '?', '...', '""', '``'] #these features can be directly extracted after tokenisation
        allcaps = 0
        excessive_punctuation = re.compile('[!?][!?]+')
        html_quotes = re.compile('&quot;')
        review_tokens = (word_tokenize(corpus_instance["TITLE"])) + corpus_instance["TOKENS"]
        review_lemmas = ((corpus_instance["TITLE"] + " " + corpus_instance["REVIEW"]))

        corpus_instance_vector = []

        for punctuation in relevant_punctuation:
            corpus_instance_vector.append(review_tokens.count(punctuation))
            #print((str(punctuation) + ": " + str(review.count(punctuation))))

        for token in review_tokens:
            if token.isupper() and len(token) > 1 and any(vowel.lower() in 'aeiuo' for vowel in token):
                allcaps += 1
        
        corpus_instance_vector.append(len(re.findall(excessive_punctuation, review_lemmas)))
        corpus_instance_vector.append(allcaps)
        corpus_instance_vector.append(len(re.findall(html_quotes, review_lemmas)))

        return corpus_instance_vector


    def get_feature_names(self):
        return ['punct_1', 'punct_2', 'punct_3', 'punct_4', 'punct_5', 
                              'punct_6', 'punct_7', 'punct_8']

if __name__ == '__main__':
    pass
                              
