import nltk
from nltk.tokenize import word_tokenize
from corpus import read_corpus

"""
These functions are meant to be accessed from training_testing.py
"""

"""
TODO:
    * get rid of tmp_list
    * vectors look plausible when tested on small corpus,
        but apparently there are up to 60 instances per bigram
        in a review (possible, but should be looked into)
"""
def extract(corpus_instance, bigram_pos_vocab):
    tmp_list = []
    tmp_list.append(corpus_instance)
    corpus_instance_pos_tagged = corpus_pos_tagger(tmp_list)
    corpus_instance_pos_unigrams = tagged_corpus_to_pos_unigrams(corpus_instance_pos_tagged)
    corpus_instance_pos_bigrams = pos_unigrams_to_bigrams(corpus_instance_pos_unigrams)
    corpus_instance_vector = []
    
    for bigram in bigram_pos_vocab:
        corpus_instance_vector.append(corpus_instance_pos_bigrams[0].count(bigram)) 
        #print(str(bigram) + ": " + str(corpus_instance_pos_bigrams[0].count(bigram)) + "\n") 
    return corpus_instance_vector


def get_pos_vocabulary(corpus):
    tagged_corpus = corpus_pos_tagger(corpus)
    pos_unigrams = tagged_corpus_to_pos_unigrams(tagged_corpus)
    pos_bigrams = pos_unigrams_to_bigrams(pos_unigrams)
    pos_vocab = to_bag_of_bigrams(pos_bigrams)

    #print stats
    print("POS Bigram vocab size:            {}".format(len(pos_vocab)))

    return pos_vocab


"""
These functions are intended for internal use.
"""

"""
Returns the raw corpus as a list 
e.g. [[('No', 'DT')], [('Just', 'RB'), ('no', 'DT')]]
"""
def corpus_pos_tagger(corpus):
    tagged_corpus = []
    temp_entry = []
    
    for entry in corpus:
        if not isinstance(entry, dict):
            continue
        
        temp_entry = nltk.pos_tag(word_tokenize(str(entry['REVIEW'])))
        tagged_corpus.append(temp_entry)
        temp_entry = []
    
    return tagged_corpus


"""
Same format as above, reduces the tuples to pos-tags
e.g. [['DT', ',', 'NN'], ['DT', ',', 'NN']]

"""
def tagged_corpus_to_pos_unigrams(tagged_corpus):
    pos_unigrams = []
    temp_pos = []
    
    for entry in tagged_corpus:
        for token in entry:
            temp_pos.append(token[1])
        pos_unigrams.append(temp_pos)
        temp_pos = []
            
    return pos_unigrams 


"""
Returns the bigrams for each review
e.g. [[('DT', ','), (',', 'NN')], [('DT', ','), (',', 'NN')]]
"""
def pos_unigrams_to_bigrams(input_list):
    bigram_list = []
    temp_bigram = []
    
    for review in input_list:
        for i in range(len(review)-1):
            temp_bigram.append((review[i], review[i+1]))
        bigram_list.append(temp_bigram)
        temp_bigram = []
           
    return bigram_list


"""
Takes all the bigrams and turns them into a bag of bigrams
e.g. [('DT', ','), (',', 'NN')]
"""
def to_bag_of_bigrams(bigram_list):
    bag_of_bigrams = []
    
    for review in bigram_list:
        for bigram in review:
            if bigram not in bag_of_bigrams:
                bag_of_bigrams.append(bigram)
                
    return bag_of_bigrams


"""
TODO: explanation that's not stupid
"""
def to_bigram_vector(bag_of_bigrams, corpus): #corpus is the bigram_list
    review_vector_list = []
    
    for entry in corpus:
        review_vector = []
        
        for bigram in bag_of_bigrams:
            review_vector.append(entry.count(bigram))
            
        review_vector_list.append(review_vector)
        
    return review_vector_list


if __name__ == '__main__':
    """
    function calls for testing purposes on a small corpus
    """
    corpus = read_corpus("minicorpus.csv")
    #for thing in corpus:
        #print(thing)
    bigram_pos_vocab = get_pos_vocabulary(corpus)
    corpus_instance = corpus[0]
    print(bigram_pos_vocab)
    print(extract(corpus_instance, bigram_pos_vocab))
    
    """
    misc. tests
    """
    #f4 = extract(corpus_instance, bigram_pos_vocab)
    #print(f4)
    #corpus_vector = to_bigram_vector(bag_of_bigrams, pos_bigrams)
    
    #for vector in corpus_vector:
        #print(vector)
        
        
"""
The functions below are intended to be used on token-level (bag of words) and are possibly obsolete
"""
#def to_token_vector(bag_of_words, corpus):
    #review_vector_list = []
    
    #for entry in corpus:
        #review_vector = []
        #review = word_tokenize(str(entry['REVIEW']))
        
        #for word in bag_of_words:
            #review_vector.append(review.count(word))
            
        #review_vector_list.append(review_vector)
        
    #return review_vector_list


#def to_bag_of_words(corpus):
    #bag_of_words = []
    
    #for entry in corpus:
        #for word in word_tokenize(str(entry['REVIEW'])):
            #if word not in bag_of_words:
                #bag_of_words.append(word)
                
    #return bag_of_words
#fun fact: len(bag_of_words) is 25325 for corpus.csv
   
