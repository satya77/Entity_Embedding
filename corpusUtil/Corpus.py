from __future__ import division

import pickle
from collections import Counter

import tensorflow as tf
from  corpusUtil.CorpusTools import CorpusTools


class Corpus(CorpusTools):
    """
    Class responsible for working with a raw text . In our case this case reads sentences from the database and orders them based on their id ,
    removes the unessecary words and keeps the entities and terms in the same order to generate a new document, this document can then be proccessed based on the
    window based approach to be used in Word2Vec
    """
    def __init__(self,setting):
        """
        :param setting: object of the setting tha has all the program properties
        """
        CorpusTools.__init__(self,setting.SAVE_MATRIX_CORPUS,setting.SAVE_FOLDERPATH,setting.CREATE_SUMMARY)
        self.corpus=[] # the final corpus in terms of a list of unique numbers
        self.corpus_path=setting.CORPUS_PATH
        self.setting=setting

    def fit_to_corpus(self):
        """
        fits a corpus object to a raw text
        :param path_to_file: the path were the text corpus is located
        :return:
        """
        print("creating the corpus object ... ")
        self.word_count=Counter()
        with open(self.corpus_path,encoding='utf-8') as f:  #read the file
            data = tf.compat.as_str(f.read()).replace("\n"," ").split(' ')


        self.word_count.update(data)
        for word, _ in self.word_count.items():# create the dictionary with unique ids
            self.all_words[word] = len(self.all_words)

        for word in data:# trun the data into unique ids instead of string
            index = self.all_words.get(word, 0)
            self.corpus.append(index)

        self.corpus=self.corpus[1:]#remove start of the file

        self.reverse_dic = dict(zip(self.all_words.values(), self.all_words.keys()))# generate the reverse dictionary
        print("***complete***\n")
        if self.save_to_file:
            self.save()



    def _findwordlables(self, client):
        if self.setting.MONGODB_AUTHENTICATION_DB != None:
            client[self.setting.MONGODB_AUTHENTICATION_DB].authenticate(self.setting.MONGODB_USER_NAME,
                                                                        self.setting.MONGODB_PASSWORD)
        db = client[self.setting.LOAD_DB]
        nodes = db[self.setting.NODES_COL]

        all_nodes = nodes.find(no_cursor_timeout=True, batch_size=500000)
        labels = {}
        for row in all_nodes:
            labels[row["type"] + "_" + str(row["id"])] = row["coverText"]
        return labels

    def save(self,path=None):
        """
        Save the trained corpus into a file for later use
        :return:
        """
        print("saving the corpus object to file ...")
        if path!=None : self.save_path=path
        pickle_out = open(str(self.save_path ) + "/trained_corpus_word2vec.pickle", "wb")
        pickle.dump(self.corpus, pickle_out)
        pickle_out.close()

        pickle_out = open(str(self.save_path ) + "/trained_all_words_word2vec.pickle", "wb")
        pickle.dump(self.all_words, pickle_out)
        pickle_out.close()

        pickel_out= open(str(self.save_path )+"/trained_reversed_dic_word2vec.pickle","wb")
        pickle.dump(self.reverse_dic,pickel_out)
        pickel_out.close()

        pickel_out = open(str(self.save_path ) + "/trained_word_counts_word2vec.pickle", "wb")
        pickle.dump(self.word_count, pickel_out)
        pickel_out.close()
        print("***complete***\n")

        # if no need for summary then we dont need to generate the label input file for tensorboard
        if self.create_summary:
            print("generating labels as input for summary ... ")
            self.generate_labels_for_embeddings()
            print("***complete***\n")

    def load(self,folder_path):
        """
        loads the trained corpus object from the file
        :param folder_path: the path that the corpus object files are located
        :return:
        """
        print("reading the corpus object from file ...")
        pickle_in = open(str(folder_path) + "/trained_corpus_word2vec.pickle", "rb")
        self.corpus = pickle.load(pickle_in)

        pickle_in = open(str(folder_path) + "/trained_all_words_word2vec.pickle", "rb")
        self.all_words = pickle.load(pickle_in)

        pickle_in=open(str(folder_path)+"/trained_reversed_dic_word2vec.pickle","rb")
        self.reverse_dic=pickle.load(pickle_in)

        pickle_in = open(str(folder_path) + "/trained_word_counts_word2vec.pickle", "rb")
        self.word_count = pickle.load(pickle_in)
        print("***complete***\n")


    def get_len_data(self):
        """
        returns the size of the corpus
        :return:
        """
        return len(self.corpus)

    def get_corpus(self):
        """
        returns the numerical corpus
        :return:
        """
        return self.corpus


