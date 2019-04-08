import numpy as np
import pandas as pd
from test.Test import Test


class WordSimilarity(Test):
    """
    Compare the methods in terms of word similarities:
    Test Collection contains two sets of English word pairs (one for training and one for testing) together with human-assigned similarity judgments,
    usually obtained by crowdsourcing, the  score reflects. According to MEN website (WordSim353 is created in the same manner):
    Rather than ask annotators to give an absolute score reflecting how much a word pair is semantically related (like in Wordsim353),
    we instead asked them to make comparative judgements on two pair exemplars at a time because this is both more natural for an individual annotator, and also permits seamless integration of the supervision from many annotators,
    each of whom may have a different internal "calibration"
    for the relatedness strengths. Moreover, binary choices were preferred because they make the construction of "right" and "wrong" control items straightforward. In total,
     each pair was rated in this way against 50 comparison pairs, thus obtaining a final score on a 50-point scale, although the Turkers' choices were binary.
    """

    def __init__(self, setting,  model=None):
        """
        Given the needed parameters for a database connection to a load network on mongodb
        :param setting: object of the setting tha has all the program properties
        :return:
        """
        Test.__init__(self, setting, model)
        self.save_path = setting.SAVE_FOLDERPATH

    def _compute_cosine(self):

        words= pd.read_csv(self.path,sep="\t",header=0,comment='#')#get all the word combinations
        #read each word and compute the cosine similarity and update the database
        for index, w in words.iterrows():
            if not self.raw_text:
                if self.model.contains(w["word1_id"]) and  self.model.contains(w["word2_id"]):
                    v1 = self.model.embedding_for(w["word1_id"])
                    v2 = self.model.embedding_for(w["word2_id"])

                    if isinstance(v1,np.ndarray) and isinstance(v2,np.ndarray) :
                       cosine=self.model.cosine_similarities(v1,v2)
                       words.at[index, self.embedding_type] = float(cosine)

            else:
                if self.model.contains(w["Word1"]) and  self.model.contains(w["Word2"]):
                    v1 = self.model.embedding_for(w["Word1"])
                    v2 = self.model.embedding_for(w["Word2"])
                    if isinstance(v1,np.ndarray) and isinstance(v2,np.ndarray) :
                        cosine=self.model.cosine_similarities(v1,v2)
                        words.at[index, self.embedding_type] = float(cosine)
        print("Pearson:")
        print(words.corr(method='pearson'))
        return words.corr(method='pearson')["Human"][self.embedding_type]


    def test_wordsim(self):
        """
        test for word similarites and print final results on the screen
        we are considering Relatedness:  the cosine similarity of the embeddings for two words should have high correlation (Spearman or Pearson)
        with human relatedness scores.
        :return:
        """

        self._compute_cosine()


    def test_wordsim_batch(self,models):
        """
        test for word similarites and print final results on the screen
        we are considering Relatedness:  the cosine similarity of the embeddings for two words should have high correlation (Spearman or Pearson)
        with human relatedness scores.
        :return:
        """
        all_corr=[]
        for m in models:
            self.model=m
            corr =self._compute_cosine()
            all_corr.append(corr)
            print(round(corr,3))
        print("Average correlatin with Human:{}".format((sum(all_corr) / len(all_corr))))

