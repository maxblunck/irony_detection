from corpus import read_corpus
from nltk.tokenize import word_tokenize

def extract(corpus_instance):
    relevant_punctuation = ['!', '?', '...', '""', '``']
    #"?!", "!?", "???", "!!!" are no lemmas
    allcaps = 0
    review = (word_tokenize(corpus_instance["TITLE"])) + corpus_instance["TOKENS"]

    corpus_instance_vector = []

    for punctuation in relevant_punctuation:
        corpus_instance_vector.append(review.count(punctuation)/len(review))
        #print((str(punctuation) + ": " + str(review.count(punctuation))))

    for token in review:
        if token.isupper() and len(token) > 1:
            allcaps += 1
    corpus_instance_vector.append(allcaps/len(review))

    return corpus_instance_vector


if __name__ == '__main__':
    """
    function calls for testing purposes on a small corpus
    """
    corpus = read_corpus("minicorpus.csv")
    corpus_instance = corpus[3]
    #print(extract(corpus_instance))
