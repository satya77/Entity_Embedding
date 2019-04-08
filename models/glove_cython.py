#!/usr/bin/env python

from collections import Counter
import  pickle

import numpy as np
import glove

from six import string_types
from gensim import utils
import pandas as pd

class GloVe(object):

    def __init__(self,corpus_path=None,vector_size=None,iterations=None,learning_rate=0.05, x_max=100, alpha=0.7,window_size=10, workers=6, min_count=None, model_number=1):
        self.embeddings=None
        self.model=None
        self.input_file=corpus_path
        self.vector_size=vector_size
        self.iterations=iterations
        self.learning_rate=learning_rate
        self.x_max=x_max
        self.alpha=alpha
        self.window_size=window_size
        self.min_count=min_count
        self.workers=workers
        self.dictionary=None
        self.model_number=model_number
        self.name="GloVe_dim={}_lr={}_itr={}_xMax={}_window={}_minCnt={}_alpha={}_num={}".format(self.vector_size,self.learning_rate,self.iterations,self.x_max,self.window_size,self.min_count,self.alpha,self.model_number)


    def read_input(self):
        """This method reads the input file which is in gzip format"""

        print("reading file {0}...this may take a while".format(self.input_file))
        word_counts = Counter()
        with open(self.input_file, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):

                if (i % 10000 == 0):
                    print("read {0} lines".format(i))
                # do some pre-processing and return a list of words for each review text
                word_counts.update(line.split(" "))

        with open(self.input_file, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                if (i % 10000 == 0):
                    print("read {0} lines".format(i))
                # do some pre-processing and return a list of words for each review text
                for i in line.split(" "):
                    if word_counts[i] < self.min_count:
                        line=line.replace(i + " ", '')
                yield line.split(" ")


    def train_all(self):
        """
        builds the vocab and trains the model
        :return:
        """
        documents = list(self.read_input())
        corpus = glove.Corpus()
        corpus.fit(documents, window=self.window_size)
        self.model = glove.Glove(no_components=self.vector_size, learning_rate=self.learning_rate, alpha=self.alpha)

        self.model.fit(corpus.matrix, epochs=self.iterations, no_threads=self.workers, verbose=True)
        self.dictionary=corpus.dictionary
        self.model.add_dictionary(corpus.dictionary)



    def save(self, path):
        self.model.save(path + "/"+self.name)
        with open(path + "/"+self.name+"_dicts.pickle", 'wb') as vector_f:
            pickle.dump(self.dictionary, vector_f, protocol=2)

    def save_to_txt(self,path,node_list_path):
        # total_vec=len(self.dictionary)
        # vector_size=self.vector_size
        # with utils.smart_open(path + "/"+self.name+".txt", 'wb') as fout:
        #     fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        #     for word,id in self.model.dictionary.items():
        #         row=self.model.word_vectors[id]
        #         fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))
        columns = ["entity_id", "cover_text"]
        self.vertext = pd.read_csv(node_list_path, sep='\t', index_col=0, header=None, comment='#')
        self.vertext.columns = columns
        total_vec = len(self.dictionary)
        vector_size = self.vector_size
        with utils.smart_open(path + "/" + self.name + ".txt", 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
            # store in sorted order: most frequent words at the top
            for  word,id in self.model.dictionary.items():
                word_label = self.vertext.loc[self.vertext['entity_id'] == word].values
                if (len(word_label) > 0):
                    word_label = word_label[0][1]
                    row = self.model.word_vectors[id]
                    fout.write(utils.to_utf8("%s %s\n" % (word_label, ' '.join("%f" % val for val in row))))



    def read_from_file(self, path):
        self.model = glove.Glove.load(path + "/"+self.name)

        pickle_in = open(path + "/"+self.name+"_dicts.pickle", "rb")
        self.dictionary = pickle.load(pickle_in)
        self.model.add_dictionary(self.dictionary)

    def embedding_for(self, words):
        """
        :param words: a word or words that embeddings are wanted.
        """

        if isinstance(words, string_types):
            return self.model.word_vectors[self.model.dictionary[words]]

        # if there are more then one word given
        all_emb = []
        for word in words:
            emb = self.model.word_vectors[self.model.dictionary[word]]

            all_emb.append(emb)
        return np.vstack(all_emb)

    def most_similar(self, positive=[], negative=[], topn=10,types=None):


        word_vec = np.zeros(self.model.word_vectors[0].shape)
        all_words=positive+negative
        for p in positive:
            if isinstance(p, string_types):

                word_vec = word_vec + self.model.word_vectors[self.model.dictionary[p]]
            else:
                word_vec = word_vec +p
        for n in negative:
            if isinstance(n, string_types):
                word_vec = word_vec - self.model.word_vectors[self.model.dictionary[n]]
                word_vec = word_vec - n

        final=[]
        if types == None:
            result = self.model._similarity_query(word_vec, topn + len(all_words))
            for r in result:
                if r[0] not in all_words:
                    final.append(r)
                    if len(final)==topn:
                        return final
        else:
            result = self.model._similarity_query(word_vec, len(self.model.dictionary)-10)
            for r in result:
                if r[0] not in all_words and r[0].split("_")[0] in types:
                    final.append(r)
                    if len(final)==topn:
                        return final
        return final


    def cosine_similarities(self, vector_1, vector_2):
        return (np.dot(vector_1, vector_2)) / np.linalg.norm(vector_1) / np.linalg.norm(vector_2)

    def contains(self, w):
        if isinstance(w, string_types):
            return w in self.dictionary.keys()
        else:
            for word in w:
                if word not in self.dictionary.keys():
                    return False

        return True

    def name(self):
        return self.name
