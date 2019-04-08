import os
import configparser

class Settings:
    """
    Class responsible for reading the setting file and providing an interface for other classes to use the setting file
    """
    def __init__(self, path_root,path_config):
        program_config = configparser.ConfigParser()
        program_config._interpolation = configparser.ExtendedInterpolation()
        program_config.read(os.path.join(path_root, path_config))
        # ----------------------------------General----------------------------------

        self.MODE = program_config.get("General",
                                  "MODE")  # Shows the current mode of the program could be "Train" to train of the models,"Test" tests the model, "Batch_Test" is
        #when mutiple models with the same parameters have been trained and we want to test them all and at once and report the avg of the test

        self.SAVE_FOLDERPATH = program_config.get("General",
                                             "SAVE_FOLDERPATH")  # the path to a folder to save the current model files
        self.EMBEDDING_TYPE = program_config.get("General",
                                            "EMBEDDING_TYPE")  # shows the type of the embedding to be trained or tested "GloVe","Word2Vec", "DeepWalk","VERSE"

        self.CORPUS_PATH = program_config.get("General",
                                         "CORPUS_PATH")  # path to the file for the textual corpus to create glove or word2vec
        self.EDGELIST_PATH =program_config.get("General","EDGELIST_PATH") # gets the path to the edge list of load

        self.NODELIST_PATH =program_config.get("General","NODELIST_PATH") # gets the path to the node list of load
        # ----------------------------------Test----------------------------------
        self.ENITY_CENTRIC_TEST =program_config.getboolean("Test","ENITY_CENTRIC_TEST") # If set is True in the test case of analogies we limit the resutls to only the entities of the same type
        self.TEST_DATA_PATH = program_config.get("Test",
                                                 "TEST_DATA_PATH")  # the location of the test dataset in form o csv files, separated by tab file

        self.TEST_MODE = program_config.get("Test",
                                            "TEST_MODE")  # Can be Clustering or Analogy or WordSimilarity

        self.TEST_BATCH_NUMBER = program_config.getint("Test",
                                            "TEST_BATCH_NUMBER")  # If we are doing batch test how many models are in the batch

        self.TEST_ON_RAW_TEXT = program_config.getboolean("Test",
                                                  "TEST_ON_RAW_TEXT")  # if set to true all the test will not use the load ids to find embeddigs but the raw form of the text, used for models trained on the raw data
# ----------------------------------Model----------------------------------
        self.NUM_EPOCH = program_config.getint("Model", "NUM_EPOCH")
        self.MODEL_NUMBER = program_config.getint("Model", "MODEL_NUMBER")# the model number defines if a model is trained multiple times and not only just once
        self.EMBEDDING_SIZE = program_config.getint("Model", "EMBEDDING_SIZE")
        self.LEARNING_RATE = program_config.getfloat("Model", "LEARNING_RATE")
        self.BATCH_SIZE = program_config.getint("Model", "BATCH_SIZE")  # size of each batch for tensorflow
        self.NUM_BATCH = program_config.getint("Model", "NUM_BATCH") # number o batches for tensorflow
        self.NUM_NEGATIVE_SAMPLES = program_config.getint("Model",
                                                 "NUM_NEGATIVE_SAMPLES")  # number of negative examples to be considered in case of word2vec
        self.WINDOW_SIZE = program_config.getint("Model",
                                            "WINDOW_SIZE")  # window size to the left and right of center word for trainig word2vec and glove
        self.PLOT_ERROR = program_config.getboolean("Model",
                                               "PLOT_ERROR")  # if set to true during training a plot of the cost function will be drawn
        self.MAX_WEIGHT_CAP = program_config.getint("Model",
                                                    "MAX_WEIGHT_CAP")  # maximum weight of the load netowrk, anything bigger than this will be caped to 1.0
        self.POWER_SCALING = program_config.getfloat("Model",
                                                     "POWER_SCALING")  # the power of the weighiting function in glove and complex, recommanded to be a fraction between 0 and 1.
        self.MIN_COUNT = program_config.getint("Model",
                                                     "MIN_COUNT")  # minum number of times a word should appear in the corpus to be accounted for in the model
        self.NUM_THREAD = program_config.getint("Model",
                                                     "NUM_THREAD")  # number of threads for the models that support multi-threading
        self.NUM_WALKS = program_config.getint("Model",
                                                  "NUM_WALKS")  # number of random walks in the deep walk based model
        self.LENGHT_WALK = program_config.getint("Model",
                                                  "LENGHT_WALK")  # lenght of random walks in the deep walk based model
        self.PPR_ALPHA = program_config.getfloat("Model",
                                                 "PPR_ALPHA")  # alpha for Personalized Page Rank in the verse model
        self.NUM_HIDDEN = program_config.getint("Model",
                                                 "NUM_HIDDEN")  # number of hidden layers in the verse model
        self.PROXIMATY = program_config.get("Model",
                                                 "PROXIMATY")
        #the proximity for verse method takes three value : "PPR", "SimRank" and "Adjacency"
        # for the DeepWalk it defines which function of the edge weights to use and can take values "Plain"-> no change to weights, "log"-> log(weight+1) and "sqrt"-> sqrt(weight)
