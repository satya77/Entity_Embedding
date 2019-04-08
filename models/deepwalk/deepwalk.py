import models.deepwalk.graph as graph
from gensim.models import Word2Vec
from models.deepwalk.skipgram import Skipgram
import random
from six import string_types
import numpy as np
import pandas as pd

class DeepWalk(object):

    def __init__(self,node_list_path,edge_list_path=None,vector_size=None,num_walks=None,length_walk=None,learning_rate=0.05,worker=6,window_size=10,negative=64,min_count=0,iterations=100,proximity="Plain",model_number=1):
        self.embeddings=None
        self.vector_size=vector_size
        self.learning_rate=learning_rate
        self.length_walk=length_walk
        self.workers=worker
        self.node_list_path=node_list_path
        self.edge_list_path=edge_list_path
        self.num_walks=num_walks
        self.negative=negative
        self.window_size=window_size
        self.min_count=min_count
        self.iterations=iterations
        self.proximity=proximity
        self.model=None
        self.model_number=model_number
        columns = ["entity_id", "cover_text"]
        self.vertext = pd.read_csv(node_list_path,index_col = 0, sep='\t', header=None, comment='#')
        self.vertext.columns = columns
        self.name="DeepWalk_dim={}_lr={}_numWalk={}_walkLen={}_window={}_neg={}_minCnt={}_iter={}_Prox={}_num={}".format(self.vector_size,self.learning_rate,self.num_walks,self.length_walk,self.window_size,self.negative,self.min_count,self.iterations,self.proximity,self.model_number)


    def train(self):
        G = graph.load_edgelist_weighted(self.edge_list_path, undirected=True,proximity=self.proximity)
        print("Number of nodes: {}".format(len(G.nodes())))

        num_walks_total = len(G.nodes()) * self.num_walks

        print("Number of walks: {}".format(num_walks_total))

        data_size = num_walks_total * self.length_walk

        print("Data size (walks*length): {}".format(data_size))

        print("Walking...")
        walks = graph.build_deepwalk_corpus(G, num_paths=self.num_walks,
                                            path_length=self.length_walk, alpha=0, rand=random.Random(),workers=self.workers)
        print("Training...")
        self.model = Word2Vec(walks, size=self.vector_size, window=self.window_size, min_count=self.min_count, negative=self.negative,
                         workers=self.workers,alpha=self.learning_rate)
        self.model.train(walks, total_examples=len(walks), epochs=self.iterations)
        self.embeddings=self.model.wv


    def save(self,path):
        self.model.save(path+"/"+self.name)

    def save_to_txt(self,path):
        self.model.wv.save_word2vec_format(path+ "/"+self.name+".txt",binary=False)

    def read_from_file(self, path):
        self.model = Skipgram.load(path +"/"+self.name)
        self.embeddings = self.model.wv


    def embedding_for(self, words):
        """
        :param words: a word or words that embeddings are wanted.
        """

        if isinstance(words, string_types):
            id_w1 = int(self.vertext.loc[self.vertext['entity_id'] == words].index[0])
            return self.model.wv[str(id_w1)]

        # if there are more then one word given
        all_emb = []
        for word in words:
            id_w1 = int(self.vertext.loc[self.vertext['entity_id'] == word].index[0])
            emb=self.model.wv[str(id_w1)]
            all_emb.append(emb)

        return np.vstack(all_emb)


    def most_similar(self, positive=[], negative=[], topn=10,types=None):
        positive_id=[]
        negative_id=[]
        for p in positive:
            if isinstance(p,string_types):
                id_w = int(self.vertext.loc[self.vertext['entity_id'] == p].index[0])
                positive_id.append(str(id_w))
            else:
                positive_id.append(p)
        for n in negative:
            if isinstance(p, string_types):
                id_w = int(self.vertext.loc[self.vertext['entity_id'] == n].index[0])
                negative_id.append(str(id_w))
            else:
                negative_id.append(n)
        result_refined = []
        if types == None:
            result=self.model.wv.most_similar(positive=positive_id, negative=negative_id, topn=topn+2)
            for r in result:
                if int(r[0]) in self.vertext.index:
                    result_refined.append((self.vertext.loc[int(r[0])].values[0],r[1]))
                if len(result_refined)>=topn:
                    return result_refined
            return result_refined
        else:
            result = self.model.wv.most_similar(positive=positive_id, negative=negative_id, topn=len(self.model.wv.vocab)-10)

            for r in result:
                if int(r[0]) in self.vertext.index:
                    text=self.vertext.loc[int(r[0])].values[0]
                    if text.split("_")[0] in types:
                        result_refined.append((text, r[1]))
                    if len(result_refined) >= topn:
                        return result_refined

    def cosine_similarities(self,vector_1, vector_2):
        return (np.dot(vector_1, vector_2)) / np.linalg.norm(vector_1) / np.linalg.norm(vector_2)


    def contains(self,w):
        if isinstance(w, string_types):
            v= self.vertext.loc[self.vertext['entity_id'] == w]
            if(len(v)<1):
                return False
            id_w = v.index[0]
            return str(id_w) in self.embeddings
        else:
            for word in w:
                id_w = self.vertext.loc[self.vertext['entity_id'] == word].index[0]
                if str(id_w) not in self.embeddings:
                    return False

        return True

    def name(self):
        return self.name



