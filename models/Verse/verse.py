import numpy as np
from gensim import matutils
from six import string_types
import pickle

class VERSE(object):
    def __init__(self,path,vector_size,edge_list_path=None):
        self.name=path.split("/")[-1]
        self.edge_list_path=edge_list_path
        self.vector_size=vector_size
        self.path=path

    def read_from_file(self):
        self.embeddings = np.fromfile(self.path + "/emb.bin", np.float32).reshape(860992, self.vector_size)
        pickle_in = open(self.path + "/dicts.pickle", "rb")
        self.node2id = pickle.load(pickle_in)
        self.id2node = {}
        for k, v in self.node2id.items():
            self.id2node[v] = k



    def embedding_for(self, words):
        """
        :param words: a word or words that embeddings are wanted.
        """

        if isinstance(words, string_types):
            id_w1 = self.node2id[words]
            return self.embeddings[id_w1]

        # if there are more then one word given
        all_emb = []
        for word in words:
            id_w1 = self.node2id[word]
            emb=self.embeddings[id_w1]
            all_emb.append(emb)

        return np.vstack(all_emb)

    def ugly_normalize(self, vecs):
        normalizers = np.sqrt((vecs * vecs).sum(axis=1))
        normalizers[normalizers == 0] = 1
        return (vecs.T / normalizers).T

    def normalize(self):
        self.embeddings = self.ugly_normalize( self.embeddings)

    def most_similar(self, positive=[], negative=[], topn=10,types=None):
        if len(positive)==1:
            dists = np.array(positive).dot(self.embeddings.T)
            if types == None:
                best = matutils.argsort(dists, topn=topn , reverse=True)
                # ignore (don't return) words from the input
                result = [(self.id2node[sim], float(dists.T[sim])) for sim in best[0] ]
            else:
                best = matutils.argsort(dists, topn=topn + 1000, reverse=True)
                # ignore (don't return) words from the input
                result = [(self.id2node[sim], float(dists.T[sim])) for sim in best[0] if
                          self.id2node[sim].split("_")[0] in types]
            return result[:topn]

        positive_id=[]
        negative_id=[]
        for p in positive:
            id_w = self.node2id[p]
            positive_id.append(id_w)
        for n in negative:
            id_w = self.node2id[n]
            negative_id.append(id_w)

        p1 =self.embeddings[positive_id[0]]
        p2 = self.embeddings[positive_id[1]]
        n1 = self.embeddings[negative_id[0]]
        # p1, p2, n1 = [(1 + self.embeddings.dot(i)) / 2 for i in (p1, p2, n1)]
        mean=(p1+p2-n1)
        dists=self.embeddings.dot(mean)

        if types == None:
            best = matutils.argsort(dists, topn=topn + 3, reverse=True)
            # ignore (don't return) words from the input
            result = [(self.id2node[sim], float(dists[sim])) for sim in best if sim not in [positive_id[0], positive_id[1], negative_id[0]]]
        else:
            best = matutils.argsort(dists, topn= topn+1000, reverse=True)
            # ignore (don't return) words from the input
            result = [(self.id2node[sim], float(dists[sim])) for sim in best if
                      sim not in [positive_id[0], positive_id[1], negative_id[0]] and  self.id2node[sim].split("_")[0] in types]
        return result[:topn]


    def cosine_similarities(self,vector_1, vector_2):
        return (np.dot(vector_1, vector_2)) / np.linalg.norm(vector_1) / np.linalg.norm(vector_2)


    def contains(self,w):
        if isinstance(w, string_types):

            return w in self.node2id.keys()
        else:
            for word in w:
                if  word in self.node2id.keys():
                    return False

        return True
