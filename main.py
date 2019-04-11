import os
import time
from datetime import timedelta
from time import gmtime
from time import strftime, localtime
from configReader.Settings import Settings
from models.Verse.verse import VERSE
from models.deepwalk.deepwalk import DeepWalk
from models.glove_cython import GloVe
from models.word2vec import Word2Vec
from test.Analogy import Analogy
from test.Clustering import Clustering

from test.WordSimilarity import WordSimilarity


def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))
#----------------------------------initialize the program ----------------------------------

print("start " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print("-----------------reading the config file ---------------------\n")
path_config="settings/settings.ini"
path_root = os.path.dirname(os.path.abspath(__file__))
setting=Settings(path_root,path_config)
print("***complete***\n")
#----------------------------------Generate coocurrence matrix----------------------------------
if setting.MODE=="Train":
    start_time = time.time()

    # #----------------------------------Train model----------------------------------
    start_time = time.time()
    print("-----------------training "+setting.EMBEDDING_TYPE+ "---------------------\n")
    model=None
   # #----------------------------------Glove model----------------------------------
    if setting.EMBEDDING_TYPE=="GloVe":
        model = GloVe(vector_size=setting.EMBEDDING_SIZE,corpus_path=setting.CORPUS_PATH,iterations=setting.NUM_EPOCH, learning_rate=setting.LEARNING_RATE,window_size=setting.WINDOW_SIZE,
                      alpha=setting.POWER_SCALING,x_max=setting.MAX_WEIGHT_CAP, min_count=setting.MIN_COUNT,workers=setting.NUM_THREAD,model_number=setting.MODEL_NUMBER)
        model.train_all()
    # #----------------------------------word2vec model----------------------------------
    elif setting.EMBEDDING_TYPE=="Word2Vec":
        model = Word2Vec(vector_size=setting.EMBEDDING_SIZE,corpus_path=setting.CORPUS_PATH, learning_rate=setting.LEARNING_RATE,iterations=setting.NUM_EPOCH,
        negative=setting.NUM_NEGATIVE_SAMPLES,window_size=setting.WINDOW_SIZE,min_count=setting.MIN_COUNT,worker=setting.NUM_THREAD,model_number=setting.MODEL_NUMBER)
        model.train()
    #----------------------------------DeepWalk model----------------------------------
    elif setting.EMBEDDING_TYPE == "DeepWalk":
        model = DeepWalk(vector_size=setting.EMBEDDING_SIZE, node_list_path=setting.NODELIST_PATH,edge_list_path=setting.EDGELIST_PATH,
                     learning_rate=setting.LEARNING_RATE, iterations=setting.NUM_EPOCH,
                     negative=setting.NUM_NEGATIVE_SAMPLES, window_size=setting.WINDOW_SIZE,
                      worker=setting.NUM_THREAD,num_walks=setting.NUM_WALKS, length_walk=setting.LENGTH_WALK,min_count=setting.MIN_COUNT,proximity=setting.PROXIMATY,model_number=setting.MODEL_NUMBER)
        model.train()
    # ----------------------------------VERSE model----------------------------------
    elif setting.EMBEDDING_TYPE == "VERSE":
       print("The VERSE model needs to be trained using the c++ package availabe in: https://github.com/xgfs/verse. For more information on how to train the model. please visit their page.")
    #----------------------------------save the trained model to file----------------------------------
    if  setting.EMBEDDING_TYPE != "VERSE":
        model.save(setting.SAVE_FOLDERPATH)

    print("---   to compelete this task ---" +secondsToStr((time.time() - start_time)))
    print("***complete***\n")
    #----------------------------------Test ----------------------------------

elif setting.MODE=="Test":
    #read the models from the file
    model=None
    if  setting.EMBEDDING_TYPE!="VERSE":
        if setting.EMBEDDING_TYPE == "GloVe":
            model = GloVe(vector_size=setting.EMBEDDING_SIZE,corpus_path=setting.CORPUS_PATH,iterations=setting.NUM_EPOCH, learning_rate=setting.LEARNING_RATE,window_size=setting.WINDOW_SIZE,
                      alpha=setting.POWER_SCALING,x_max=setting.MAX_WEIGHT_CAP, min_count=setting.MIN_COUNT,workers=setting.NUM_THREAD,model_number=setting.MODEL_NUMBER)
        elif setting.EMBEDDING_TYPE == "Word2Vec":
            model = Word2Vec(vector_size=setting.EMBEDDING_SIZE,corpus_path=setting.CORPUS_PATH, learning_rate=setting.LEARNING_RATE,iterations=setting.NUM_EPOCH,
        negative=setting.NUM_NEGATIVE_SAMPLES,window_size=setting.WINDOW_SIZE,min_count=setting.MIN_COUNT,worker=setting.NUM_THREAD,model_number=setting.MODEL_NUMBER)
        elif setting.EMBEDDING_TYPE == "DeepWalk":
            model = DeepWalk(vector_size=setting.EMBEDDING_SIZE, node_list_path=setting.NODELIST_PATH,edge_list_path=setting.EDGELIST_PATH,
                     learning_rate=setting.LEARNING_RATE,
                     negative=setting.NUM_NEGATIVE_SAMPLES, window_size=setting.WINDOW_SIZE,
                      worker=setting.NUM_THREAD,num_walks=setting.NUM_WALKS, iterations=setting.NUM_EPOCH, length_walk=setting.LENGTH_WALK,min_count=setting.MIN_COUNT,proximity=setting.PROXIMATY,model_number=setting.MODEL_NUMBER)
        model.read_from_file(setting.SAVE_FOLDERPATH)

    elif setting.EMBEDDING_TYPE == "VERSE":
        model=VERSE(path=setting.SAVE_FOLDERPATH,vector_size=setting.EMBEDDING_SIZE)
        model.read_from_file()
        if setting.TEST_MODE == "Analogy":
            model.normalize()
    start_time = time.time()

    if setting.TEST_MODE=="Clustering":
        print("----------------- Creating clusters for "+setting.EMBEDDING_TYPE+"---------------------\n")
        clustering = Clustering(setting, model=model)
        clustering.create_clusters(path="./")

    elif setting.TEST_MODE == "Analogy":
        print("----------------- Testing Analgoies for "+setting.EMBEDDING_TYPE+"---------------------\n")
        analogy = Analogy(setting, model=model,enitity_centric=setting.ENITY_CENTRIC_TEST)
        analogy.test_analogies()
    elif setting.TEST_MODE == "WordSimilarity":
        print("----------------- Testing word similarities for "+setting.EMBEDDING_TYPE+"---------------------\n")
        wordsim = WordSimilarity(setting, model=model)
        wordsim.test_wordsim()

    # ----------------------------------Batch Test ----------------------------------

elif setting.MODE == "Batch_Test":
    # read the models from the file
    models=[]
    for num in range(1,setting.TEST_BATCH_NUMBER+1):
        model = None
        if  setting.EMBEDDING_TYPE != "VERSE":
            if setting.EMBEDDING_TYPE == "GloVe":
                model = GloVe(vector_size=setting.EMBEDDING_SIZE, corpus_path=setting.CORPUS_PATH,
                              iterations=setting.NUM_EPOCH, learning_rate=setting.LEARNING_RATE,
                              window_size=setting.WINDOW_SIZE,
                              alpha=setting.POWER_SCALING, x_max=setting.MAX_WEIGHT_CAP, min_count=setting.MIN_COUNT,
                              workers=setting.NUM_THREAD,model_number=num)
            elif setting.EMBEDDING_TYPE == "Word2Vec":
                model = Word2Vec(vector_size=setting.EMBEDDING_SIZE, corpus_path=setting.CORPUS_PATH,
                                 learning_rate=setting.LEARNING_RATE, iterations=setting.NUM_EPOCH,
                                 negative=setting.NUM_NEGATIVE_SAMPLES, window_size=setting.WINDOW_SIZE,
                                 min_count=setting.MIN_COUNT, worker=setting.NUM_THREAD,model_number=num)
            elif setting.EMBEDDING_TYPE == "DeepWalk":
                model = DeepWalk(vector_size=setting.EMBEDDING_SIZE, node_list_path=setting.NODELIST_PATH,
                                 edge_list_path=setting.EDGELIST_PATH,
                                 learning_rate=setting.LEARNING_RATE,
                                 negative=setting.NUM_NEGATIVE_SAMPLES, window_size=setting.WINDOW_SIZE,
                                 worker=setting.NUM_THREAD, num_walks=setting.NUM_WALKS, iterations=setting.NUM_EPOCH,
                                 length_walk=setting.LENGTH_WALK, min_count=setting.MIN_COUNT,
                                 proximity=setting.PROXIMATY,model_number=num)
            model.read_from_file(setting.SAVE_FOLDERPATH)
        elif setting.EMBEDDING_TYPE == "VERSE":
            model = VERSE(path=setting.SAVE_FOLDERPATH+"_num="+str(num), vector_size=setting.EMBEDDING_SIZE)
            model.read_from_file(setting.SAVE_FOLDERPATH)
            if setting.TEST_MODE == "Analogy":
                model.normalize()
        models.append(model)
    start_time = time.time()
    
    if setting.TEST_MODE == "Clustering":
        print("----------------- creating clusters ---------------------\n")
        clustering = Clustering(setting, model=model)
        clustering.create_clusters_batch(models=models)
    elif setting.TEST_MODE == "Analogy":
        print("----------------- Testing Analgoies ---------------------\n")
        analogy = Analogy(setting, model=None, enitity_centric=setting.ENITY_CENTRIC_TEST)
        analogy.test_analogies_batch(models=models)
    elif setting.TEST_MODE == "WordSimilarity":
        print(setting.SPECIFIC_TYPE)
        print(
            "----------------- Testing word similarities ---------------------\n")
        wordsim = WordSimilarity(setting, model=None)
        wordsim.test_wordsim_batch(models=models)

    print("---   to compelete this task ---" + secondsToStr((time.time() - start_time)))
    print("***complete***\n")

