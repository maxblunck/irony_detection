from corpus import read_corpus
from nltk.tokenize import word_tokenize
import re

def extract(corpus_instance):
    relevant_punctuation = ['!', '?', '...', '""', '``']
    #"?!", "!?", "???", "!!!" are no lemmas
    allcaps = 0
    excessive_punctuation = re.compile('[!?][!?]+')
    review_tokens = (word_tokenize(corpus_instance["TITLE"])) + corpus_instance["TOKENS"]
    review_lemmas = ((corpus_instance["TITLE"] + " " + corpus_instance["REVIEW"]))

    corpus_instance_vector = []

    for punctuation in relevant_punctuation:
        corpus_instance_vector.append(review_tokens.count(punctuation)/len(review_tokens))
        #print((str(punctuation) + ": " + str(review.count(punctuation))))

    for token in review_tokens:
        if token.isupper() and len(token) > 1 and any(vowel.lower() in 'aeiuo' for vowel in token):
            allcaps += 1
    
    corpus_instance_vector.append(len(re.findall(excessive_punctuation, review_lemmas))/len(review_lemmas))
    corpus_instance_vector.append(allcaps/len(review_tokens))


    return corpus_instance_vector


if __name__ == '__main__':
    """
    function calls for testing purposes on a small corpus
    """
    pass
    #corpus = read_corpus("minicorpus.csv")
    #corpus_instance = corpus[3]
    #print(extract(corpus_instance))
