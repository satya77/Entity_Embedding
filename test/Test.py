class Test(object):
    """
    Base class for all the other classes that do the testing
    """
    def __init__(self,setting,model=None):
        """
        Given the needed parameters for a database connection to a load network on mongodb
        :return:
        """
        self.model = model
        self.embedding_type = setting.EMBEDDING_TYPE
        self.path=setting.TEST_DATA_PATH
        self.raw_text = setting.TEST_ON_RAW_TEXT

