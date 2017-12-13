import nltk
from nltk.tokenize import word_tokenize
from corpus import read_corpus

"""
turning the entire corpus into a bag of words (lemmas).
returns: list
"""
def to_bag_of_words(corpus):
    for entry in corpus:
        for word in word_tokenize(str(entry['REVIEW'])):
            if word not in bag_of_words:
                bag_of_words.append(word)
    return bag_of_words


"""
pos-tagging the entire corpus token-wise.
"""
def corpus_pos_tagger(corpus):
    for entry in corpus:
        tagged_corpus.append(nltk.pos_tag(word_tokenize(str(entry['REVIEW']))))
    return tagged_corpus


"""
for each review in the corpus, the number of occurences of each token is written
into a feature vector of the same length as the bag of words list.
returns: list of lists
"""
def to_vector(bag_of_words, corpus):
    sentence_vector_list = []

    for entry in corpus:
        sentence_vector = []
        review = word_tokenize(str(entry['REVIEW']))

        for word in bag_of_words:
            sentence_vector.append(review.count(word))

        sentence_vector_list.append(sentence_vector)

    return sentence_vector_list


if __name__ == '__main__':
    corpus = read_corpus("minicorpus.csv")
    bag_of_words = []
    tagged_corpus = []

    bag_of_words = to_bag_of_words(corpus)

    #das sollte rausgenommen werden, wenn mit dem kompletten korpus gearbeitet wird
    for vektor in to_vector(bag_of_words, corpus):
        print (str(vektor) + "\n")

    if len(bag_of_words) != len(to_vector(bag_of_words, corpus)[0]):
        print("Irgendwas lief schief (Featurevektor und Bag of Words nicht gleich lang)")
