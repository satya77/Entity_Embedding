

import gensim
from six import string_types, iteritems
import numpy as np
import pandas as pd
from gensim import utils

class Word2Vec(object):


    def __init__(self,corpus_path=None,vector_size=None,iterations=None,learning_rate=0.05,window_size=10, min_count=None,worker=6,negative=64,model_number=2):
        self.corpus_path=corpus_path
        self.vector_size=vector_size
        self.iterations=iterations
        self.learning_rate=learning_rate
        self.window_size=window_size
        self.min_count=min_count
        self.workers=worker
        self.negative=negative
        self.embeddings=None
        self.model=None
        self.model_number=model_number
        self.name="Word2Vec_dim={}_lr={}_itr={}_window={}_minCnt={}_neg={}_num={}".format(self.vector_size,self.learning_rate,self.iterations,self.window_size,self.min_count,self.negative,self.model_number)


    def read_input(self):
        """This method reads the input file for the corpus """

        print("reading file {0}...this may take a while".format(self.corpus_path))

        with open(self.corpus_path, 'r',encoding='utf-8') as f:
            for i, line in enumerate(f):

                if (i % 10000 == 0):
                    print("read {0} lines".format(i))
                # do some pre-processing and return a list of words for each review text
                yield line.split(" ")

    def train(self):
        documents = list(self.read_input())
        print("Trianing... ")
        m= gensim.models.Word2Vec(documents, size=self.vector_size, window=self.window_size, negative=self.negative, min_count=self.min_count, alpha=self.learning_rate, workers=self.workers)
        m.train(documents, total_examples=len(documents), epochs=self.iterations)
        self.embeddings=m.wv
        self.model=m
        return m

    def save(self,path):
        self.model.save(path+ "/"+self.name)

    def save_to_txt(self,path,node_list_path):
        columns = ["entity_id", "cover_text"]
        self.vertext = pd.read_csv(node_list_path, sep='\t', index_col=0, header=None, comment='#')
        self.vertext.columns = columns
        vector_size = self.model.wv.vectors.shape[1]
        total_vec=len(self.model.wv.vocab)
        with utils.smart_open(path+ "/"+self.name+".txt", 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
            # store in sorted order: most frequent words at the top
            for word, vocab_ in sorted(iteritems(self.model.wv.vocab), key=lambda item: -item[1].count):

                word_label=self.vertext.loc[self.vertext['entity_id'] == word].values
                if(len(word_label)>0):
                    word_label=word_label[0][1]
                    row = self.model.wv.vectors[vocab_.index]
                    fout.write(utils.to_utf8("%s %s\n" % (word_label, ' '.join("%f" % val for val in row))))

        # self.model.wv.save_word2vec_format(path+ "/"+self.name+".txt",binary=False)

    def read_from_file(self,path):
        self.model=gensim.models.Word2Vec.load(path+ "/"+self.name)
        self.embeddings=self.model.wv


    def embedding_for(self, words):
        """
        :param words: a word or words that embeddings are wanted.
        """
        if isinstance(words, string_types):
                return self.model[words]

        # if there are more then one word given
        all_emb = []
        for word in words:
            emb=self.model[word]

            all_emb.append(emb)
        return np.vstack(all_emb)

    def most_similar(self, positive=None, negative=None, topn=1,types=None):
        if types==None:
            return  self.model.wv.most_similar(positive=positive, negative=negative, topn=topn)
        all_result=[]
        for i in self.model.wv.most_similar(positive=positive, negative=negative, topn=len(self.model.wv.vocab)-10):
            if i[0].split("_")[0] in types:
                all_result.append(i)
                if len(all_result)==topn:
                    return all_result


    def cosine_similarities(self,vector_1, vector_2):
        return (np.dot(vector_1, vector_2)) / np.linalg.norm(vector_1) / np.linalg.norm(vector_2)

    def contains(self,w):
        if isinstance(w, string_types):
            return w in self.embeddings
        else:
            for word in w:
                if word not in self.embeddings:
                    return False

        return True

    def name(self):
        return self.name

