from interact import predict
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys

wikifile = sys.argv[1]#"wiki_10000.pkl"
invfile = sys.argv[2]#"inverted_index_10000.pkl"
pathfile = sys.argv[3]#"/Users/skosgi/Downloads/drqanetwork"
question = sys.argv[4]#"Where does india stand in troop contributor united nations?"
vocab_path =  sys.argv[5]#"vocab_set"


wiki = pickle.load(open(wikifile,'rb'))
inverted_index = pickle.load(open(invfile,'rb'))


class DocumentRetriever:
    def __init__(self):
        return
    # pass question as q
    def find(self, q='What is anarchism?', count=5):
        # split into words
        q = q.lower().translate(str.maketrans('', '', string.punctuation)).split()

        # inverted-index lookup
        titles = []
        for q_word in q:
            titles.extend(inverted_index[q_word])
        titles = list(set(titles))

        # prepare corpus of articles found in inverted-index lookup
        question = ' '.join(q)
        corpus = [question]
        for (index, title) in titles:
            text = wiki.iloc[index, 0].upper() + ' | ' + wiki.iloc[index, 1]
            corpus.append(str(text))

        # TF-IDF on articles found in inverted-index lookup
        tfidf = TfidfVectorizer()
        vecs = tfidf.fit_transform(corpus)
        corr_matrix = ((vecs * vecs.T).A)
        result = [corpus[i] for i in np.argsort(corr_matrix[0])[::-1][:count+1]]

        return result[1:]


ret = DocumentRetriever()

contexts = ret.find(q=question, count=2)


for context in contexts:
    print(predict(question,context,pathfile,"RNN",vocab_path))
