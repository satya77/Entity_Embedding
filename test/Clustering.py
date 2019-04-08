import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from test.Test import Test
from sklearn import cluster
import warnings
from itertools import cycle, islice
from sklearn.metrics import accuracy_score

class Clustering(Test):
    """
    Class responsible for a both visual and numerical comparison between the methods in terms of clustering, given a collection with tagged groups of words
    T-sne plot of the clusters are generated, It is also possible to measure the clusters in terms of supervised clustering with Rand-Index , Precision, Recall ...
    """

    def __init__(self,setting,model=None):
        """
        Given the needed parameters for a database connection to a load network on mongodb
        :param setting: object of the setting tha has all the program properties
        :return:
        """
        Test.__init__(self,setting, model)
        self.words=pd.read_csv(self.path,sep="\t",header=0,comment='#')#get all the word combinations
        self.categories=list(self.words.category.unique())
        self.color_list={}
        counter=0
        for s in self.categories:
            counter+=1
            self.color_list[s]=counter


    def _fill_embeddings_for_clustering(self):
        labels = []
        colors=[]
        cats=[]
        cats_ids=[]
        cats_to_ids = {}
        for i, c in enumerate(self.categories):
            cats_to_ids[c] = i

        all_emb = []
        if not self.raw_text:
            for index,w in self.words.iterrows():
                if self.model.contains(w["unique_key"]):
                    cats.append(w['category'])

                    all_emb.append(self.model.embedding_for(w["unique_key"]))
                    cats_ids.append(cats_to_ids[w['category']])
                    colors.append(self.color_list[w["category"]])
                    labels.append(w['label'])
        else:
            for index,w in self.words.iterrows():
                parts=w["label"].split(" ")
                if len(parts)>1:
                    mean=np.zeros(self.model.vector_size)
                    count=0
                    for p in parts:
                        if self.model.contains(p.lower()):
                            mean=mean+self.model.embedding_for(p.lower())
                            count=count+1
                        if count ==len(parts):
                            mean=mean/count
                            all_emb.append(mean)
                            cats.append(w['category'])
                            cats_ids.append(cats_to_ids[w['category']])
                            colors.append(self.color_list[w["category"]])
                            labels.append(w['label'])
                else:
                    if self.model.contains(w["label"].lower()):
                        all_emb.append(self.model.embedding_for(w["label"].lower()))
                        cats.append(w['category'])
                        cats_ids.append(cats_to_ids[w['category']])
                        colors.append(self.color_list[w["category"]])
                        labels.append(w['label'])
        return labels, np.vstack(all_emb),colors,cats,cats_ids


    def purity_score(self,y_true, y_pred):
        """Purity score

        To compute purity, each cluster is assigned to the class which is most frequent
        in the cluster [1], and then the accuracy of this assignment is measured by counting
        the number of correctly assigned documents and dividing by the number of documents.
        We suppose here that the ground truth labels are integers, the same with the predicted clusters i.e
        the clusters index.

        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score

        References:
            [1] https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
        """
        # matrix which will hold the majority-voted labels
        y_voted_labels = np.zeros(y_true.shape)
        # Ordering labels
        ## Labels might be missing e.g with set like 0,2 where 1 is missing
        ## First find the unique labels, then map the labels to an ordered set
        ## 0,2 should become 0,1
        labels = np.unique(y_true)
        ordered_labels = np.arange(labels.shape[0])
        for k in range(labels.shape[0]):
            y_true[y_true == labels[k]] = ordered_labels[k]
        # Update unique labels
        labels = np.unique(y_true)
        # We set the number of bins to be n_classes+2 so that
        # we count the actual occurence of classes between two consecutive bin
        # the bigger being excluded [bin_i, bin_i+1[
        bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

        for cluster in np.unique(y_pred):
            hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
            # Find the most present label in the cluster
            winner = np.argmax(hist)
            y_voted_labels[y_pred == cluster] = winner

        return accuracy_score(y_true, y_voted_labels)

    def create_clusters(self,path="./"):
        labels, embeddings, colors,_,cats = self._fill_embeddings_for_clustering()

        two_means = cluster.MiniBatchKMeans(init='k-means++',n_clusters=len(self.categories))
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cosine",
            n_clusters=len(self.categories))


        clustering_algorithms = (
            ('MiniBatchKMeans', two_means),
            ('AgglomerativeClustering', average_linkage)
        )
        plot_num = 1
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings)
        figure = plt.figure(figsize=(18,7), facecolor="white", edgecolor="black")
        plt.subplots_adjust(left=.02, right=.98, bottom=.01, top=.96, wspace=.05,
                            hspace=.11)
        for name, algorithm in clustering_algorithms:

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():

                algorithm.fit(embeddings)

                if hasattr(algorithm, 'labels_'):
                    cluster_labels = algorithm.labels_.astype(np.int)
                else:
                    cluster_labels = algorithm.predict(embeddings)
                print()
                print("results for :{}".format(name))
                purity=self.purity_score(np.array(cats), np.array(cluster_labels))
                print("Purity:{}".format(purity))

                plt.subplot(1, 3, plot_num)
                plt.title(name, size=18)

                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']),
                                              int(max(cluster_labels) + 2))))
                plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], s=70, color=colors[cluster_labels])

                plt.xticks(())
                plt.yticks(())
                plot_num += 1

        plt.subplot(1, 3, plot_num)
        plt.title("True labels ", size=18)

        plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], s=70, color=colors[cats])
        plt.xticks(())
        plt.yticks(())
        figure.savefig(path+"/clusters_"+self.model.name+"_"+self.path.split("/")[-1]+".png")
        return purity

    def plot_with_labels(self,low_dim_embs, labels, path, size,category,colors,cats):
        """
        given the lower dim representations and their labels creates a plot of embeddings

        Arguments:
        :param low_dim_embs : lower dimensional embeddings
        :param labels : labels for the embeddings
        :param path : the path to save the plot if None it will not be saved and only shown
        :param size : the size of the image
        :param category: the categories that are shown in the plot
        """
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        #
        # palette= np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
        #                                              '#f781bf', '#a65628', '#984ea3',
        #                                              '#999999', '#e41a1c', '#dede00']),
        #                                        num_clusters+2)))
        palette = np.array(['#377eb8','#F39038', '#07FE1C', '#7D3E95','#377eb8', '#ff7f00', '#4daf4a','#377eb8', '#ff7f00', '#4daf4a','#377eb8'])
        # palette =np.array(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'])
        category_copy=category.copy()
        figure = plt.figure(figsize=size,facecolor="white",edgecolor="black")  # in inches
        axes = figure.add_subplot(111)
        axes.patch.set_facecolor("white")
        axes.grid(color='lightgrey',alpha=0.3)
        axes.spines['right'].set_color('red')
        axes.spines['left'].set_color('red')
        axes.axis('off')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            color=colors[i]
            if cats[i] in category_copy:
                plt.scatter(x, y, s=150,c=palette[color],label=cats[i],alpha=1)
                category_copy.remove(cats[i])
            else:
                plt.scatter(x, y, s=150, c=palette[color],alpha=1)
        # plt.legend()
        plt.close(figure)


    def create_clusters_batch(self, models):
        all_purity={'MiniBatchKMeans':[],'AgglomerativeClustering':[]}

        two_means = cluster.MiniBatchKMeans(init='k-means++', n_clusters=len(self.categories))
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cosine",
            n_clusters=len(self.categories))

        clustering_algorithms = (
            ('MiniBatchKMeans', two_means),
            ('AgglomerativeClustering', average_linkage)
        )
        for name, algorithm in clustering_algorithms:
            print(name)
            for m in models:
                self.model = m
                labels, embeddings, colors, _, cats = self.get_embeddings_and_labels()

                algorithm.fit(embeddings)

                if hasattr(algorithm, 'labels_'):
                    cluster_labels = algorithm.labels_.astype(np.int)
                else:
                    cluster_labels = algorithm.predict(embeddings)
                    purity = self.purity_score(np.array(cats), np.array(cluster_labels))
                all_purity[name].append(purity)
                print(round(purity,3))
        print("Averrage Purity for Kmeans: {} for Agg: {}".format((sum(all_purity['MiniBatchKMeans'])/len(all_purity['MiniBatchKMeans'])),(sum(all_purity['AgglomerativeClustering'])/len(all_purity['AgglomerativeClustering']))))


