from corpus import read_corpus

corpus = read_corpus("corpus.csv")

print(len(corpus))

print(corpus[41])

print(corpus[41].keys())

print(corpus[41]["TITLE"])
print(corpus[41]["STARS"])