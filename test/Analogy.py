
from test.Test import Test
import pandas as pd


class Analogy(Test):
    """
        Compare the methods in terms of analogy task defined by Mikolove in 2013 :
        To find a word that is similar to small in the same sense as
        biggest is similar to big, we can simply compute vector X = vector(”biggest”)−vector(”big”) +
        vector(”small”). Then, we search in the vector space for the word closest to X measured by cosine
        distance. When the word vectors are well trained, it is possible to find the correct answer (word
        smallest) using this method.
    """
    def __init__(self,setting,model=None,enitity_centric=False):
        """
        Given the needed parameters for a database connection to a load network on mongodb
        :param setting: object of the setting tha has all the program properties
        :return:
        """
        Test.__init__(self, setting,model)
        self.num_total=0
        self.wrong_list={}
        self.correct_list={}
        self.type_to_wordset={}#key type
        self.save_path=setting.SAVE_FOLDERPATH
        self.entity_centric=enitity_centric
        self.words=pd.read_csv(self.path,sep="\t",header=0,comment='#')#get all the word combinations


    def _find_types(self):
        self.analoy_types= self.words.type.unique()
        for t in self.analoy_types:
            self.type_to_wordset [t]={}

    def _find_embeddings(self):
        #fill in the dictionary with all the word pairs
        if not self.raw_text:
            for index,w in self.words.iterrows():
                self.type_to_wordset[w['type']][(w["set1_word1"],w["set1_word2"],w["set2_word1"],w["set2_word2"])]=(w["set1_word1_id"],w["set1_word2_id"],w["set2_word1_id"],w["set2_word2_id"])
        else:
            for index,w in self.words.iterrows():
                self.type_to_wordset[w['type']][
                    (w["set1_word1"].lower(), w["set1_word2"].lower(), w["set2_word1"].lower(), w["set2_word2"].lower())] = (
                w["set1_word1"].lower(), w["set1_word2"].lower(), w["set2_word1"].lower(), w["set2_word2"].lower())
        return self.words.count()

    def find_closest(self):
        for t in self.analoy_types:
                correct = 0#counter for each type
                wrong = 0
                total_type=len(self.type_to_wordset[t])

                for w,key in self.type_to_wordset[t].items():
                    if(self.model.contains(key[0]) and self.model.contains(key[1])and self.model.contains(key[2])and self.model.contains(key[3])):
                        if self.entity_centric:
                            types=[key[0].split("_")[0],key[1].split("_")[0],key[2].split("_")[0]]

                        else:
                            types = None
                        result = self.model.most_similar(positive=[key[0], key[2]], negative=[key[1]], topn=1,
                                                         types=types)  # find the most similiar word
                        prediction=result[0]
                        if prediction[0]==key[3]:
                            self.correct_list[w]=prediction[0]
                            correct=correct+1
                        else:
                            self.wrong_list[w]=prediction[0]
                            wrong=wrong+1

                        self.num_total=self.num_total+1
                print("Accuracy for type {}:{} , {} out of {} were correct. and {} wrong.".format(t, float(
                    correct) / total_type, correct,total_type, wrong))
        acc=float(len(self.correct_list))/self.num_total
        print("Accuracy :{} , {} out of {} were correct. and {} wrong.".format(acc,len(self.correct_list),self.num_total,len(self.wrong_list)))
        return acc

    def test_analogies(self):
        """
        test for analgoies and print final results on the screen
        :return:
        """
        self._find_types()
        total=self._find_embeddings()
        self.find_closest()


    def test_analogies_batch(self,models):
        """
        test for analgoies and print final results on the screen
        :return:
        """
        all_acc = []
        for m in models:
            self.model = m
            self._find_types()
            total = self._find_embeddings()
            acc=self.find_closest()
            all_acc.append(acc)
            print(acc)
        print("Average Accuracy with Human:{}".format(sum(all_acc) / len(all_acc)))




