import nltk
from nltk.tokenize import word_tokenize
from corpus import read_corpus

corpus = read_corpus("corpus.csv")
tagged_corpus = []

# for debugging purposes. if you're sure it's worth it, use
# for i in range(len(corpus)):
for i in range (9):
    tagged_corpus.append(nltk.pos_tag(word_tokenize(str(corpus[i]['REVIEW']))))

print (tagged_corpus)
