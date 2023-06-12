import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenization(sentence):
    return nltk.word_tokenize(sentence)

def stemming(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokens,all_words):
    bag = np.zeros(len(all_words),dtype=np.float32)
    for i,word in enumerate(all_words):
        stemmed_word = stemming(word)
        if stemmed_word in tokens:
            bag[i] = 1.0
    
    return bag