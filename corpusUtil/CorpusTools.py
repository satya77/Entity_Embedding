from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder
class CorpusTools(object):
    """
        The main class for all the corpus related work, this can be creation of a coocurrence matrix or raw text corpus
        """

    def __init__(self,save_to_file=True, save_path="./resources",create_summary=False,setting=None):
        self.all_words = {}  # map of words(strings) to their codes(integers)
        self.reverse_dic = {}  # maps codes(integers) to words(strings)
        self.word_count = None  # map of words(strings) to count of occurrences
        # in case of a coocurrence matrix this is no longer simple counts:
        # map of words(strings) to sum of their weights this can be the number of occurrences in
        # a simiple co-ocurrence matrix or the sum of the weights of LOAD graph
        self.save_to_file = save_to_file
        self.save_path = save_path
        self.create_summary=create_summary
        self.setting=setting



    def get_all_words_to_id(self):
        """
        get the dictionary of all words to its unique number

        Returns:
        :return:all the words in the vocaburary with its unique numbers
        """
        return self.all_words

    def get_all_ids_to_words(self):
        """
        get the dictionary of all the ids and words

        Returns:
        :return:all the words in the vocaburary with its unique numbers
        """
        return self.reverse_dic

    def get_vocab_size(self):
        """
         get the  vocab size

        Returns:
        :return:number of unique words in the  vocab
        """
        return len(self.all_words)

    def get_vocab(self):
        """
        get all the words in the vocab
        :return:
        list of all the words in vocab
        """
        return list(self.all_words.keys())

    def get_vocab_ids(self):
        """
        returns all the ids
        :return: the id
        """
        return self.all_words.values()

    def get_word_for_id(self, id):
        """
        return the word for an id
        :param id: the id
        :return: the word
        """
        return self.reverse_dic[id]

    def get_id_for_word(self, word):
        """
        return the id for an word
        :param word: the id
        :return: the id
        """
        if word not in self.all_words:
            return None
        return self.all_words[word]

    def get_most_common_words(self, n):
        """
        get the top words based on their counts
        :param n: number of commons to be returns
        :return: top n common words
        """
        return [x[0] for x in self.word_count.most_common(n)]

    def get_most_common_ids(self, n):
        """
        get the top ids of words based on their counts
        :param n: number of commons to be returns
        :return: top n common ids
        """
        return [self.get_id_for_word(x[0]) for x in self.word_count.most_common(n)]

    def get_all_words(self):
        """
        return all the words in vocabulary
        :return: all the words in vocan
        """
        return self.all_words.keys()

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

    def generate_labels_for_embeddings(self):
        """
        in case we want a summary to be generated for the embedding we will need a file that contains the metadata or the corresponding labels for each node
        this helps for the visualization in tensorboard.
        :return:
        """
        if not self.setting.ON_SERVER:
            with SSHTunnelForwarder((self.setting.SSH_HOST, 22), ssh_username=self.setting.LDAP_USER_NAME,
                                    ssh_password=self.setting.LDAP_PASSWORD,
                                    remote_bind_address=(self.setting.MONGODB_HOST, self.setting.MONGODB_PORT),
                                    local_bind_address=('localhost', self.setting.MONGODB_PORT)) as server:
                client = MongoClient('localhost', self.setting.MONGODB_PORT)
                try:
                    labels = self._findwordlables(client)
                finally:
                    client.close()


        else:
            client = MongoClient(self.setting.MONGODB_HOST, self.setting.MONGODB_PORT)
            try:
                labels = self._findwordlables(client)
            finally:
                client.close()

        s = sorted(self.reverse_dic.items())
        with open(self.save_path + "/metadata.tsv", "wb") as record_file:
            for id, word in s:
                if word == '':
                    record_file.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
                else:
                    if word in labels.keys():
                        record_file.write("{0}".format(labels[word]).encode('utf-8') + b'\n')
                    else:
                        record_file.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')