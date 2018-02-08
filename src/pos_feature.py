from feature import Feature
import nltk
from nltk.tokenize import word_tokenize
import config


class PosFeature(Feature):
    """
    Class representing feature f2

    extract-method returns a feature-vector of length of its vocabulary 
    containing pos-bigram counts

    """

    vocabulary = None


    def extract(self, corpus_instance):
        tmp_list = []
        tmp_list.append(corpus_instance)
        corpus_instance_pos_tagged = self.__corpus_pos_tagger(tmp_list)
        corpus_instance_pos_unigrams = self.__tagged_corpus_to_pos_unigrams(corpus_instance_pos_tagged)
        corpus_instance_pos_bigrams = self.__pos_unigrams_to_bigrams(corpus_instance_pos_unigrams)
        corpus_instance_vector = []
        
        for bigram in self.vocabulary:
            corpus_instance_vector.append(corpus_instance_pos_bigrams[0].count(bigram)) 
            #print(str(bigram) + ": " + str(corpus_instance_pos_bigrams[0].count(bigram)) + "\n") 

        return corpus_instance_vector


    def load_vocabulary(self, corpus):
        tagged_corpus = self.__corpus_pos_tagger(corpus)
        pos_unigrams = self.__tagged_corpus_to_pos_unigrams(tagged_corpus)
        pos_bigrams = self.__pos_unigrams_to_bigrams(pos_unigrams)
        pos_vocab = self.__to_bag_of_bigrams(pos_bigrams)

        self.vocabulary = pos_vocab

        if config.print_stats == True:
            print("Bag-of-POS Vocab size (n=2):\t\t{}".format(len(self.vocabulary)))


    def get_feature_names(self):
        return [str(x) for x in self.vocabulary]


    """
    Returns the raw corpus as a list 
    e.g. [[('No', 'DT')], [('Just', 'RB'), ('no', 'DT')]]
    """
    def __corpus_pos_tagger(self, corpus):
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
    def __tagged_corpus_to_pos_unigrams(self, tagged_corpus):
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
    def __pos_unigrams_to_bigrams(self, input_list):
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
    def __to_bag_of_bigrams(self, bigram_list):
        bag_of_bigrams = []
        
        for review in bigram_list:
            for bigram in review:
                if bigram not in bag_of_bigrams:
                    bag_of_bigrams.append(bigram)
                    
        return bag_of_bigrams


    """
    Counts the occurrences of each bigram from the bag of bigrams for each review
    """
    def __to_bigram_vector(self, bag_of_bigrams, corpus): #corpus is the bigram_list
        review_vector_list = []
        
        for entry in corpus:
            review_vector = []
            
            for bigram in bag_of_bigrams:
                review_vector.append(entry.count(bigram))
                
            review_vector_list.append(review_vector)
            
        return review_vector_list
    

if __name__ == '__main__':
    pass
