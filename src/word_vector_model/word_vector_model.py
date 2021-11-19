from gensim.models import KeyedVectors
import gensim.downloader as api


class WordVectorModel:
    MODEL_NAME = 'word2vec-google-news-300'

    def __init__(self):
        self.model = self.select_initialized_model()

    def select_initialized_model(self):
        wv = KeyedVectors.load(self.MODEL_NAME)
        if wv: return wv

        wv = api.load(self.MODEL_NAME)
        wv.save(self.MODEL_NAME)
        return wv
