from gensim.models import KeyedVectors
import gensim.downloader as api


class WordVectorModel:
    MODEL_NAME = 'word2vec-google-news-300'

    def __init__(self):
        self.model = self.select_initialized_model()
        self.name = self.MODEL_NAME
        self.vocabulary_size = len(list(self.model.index_to_key))
        self.valid_answer_count = 0
        self.non_guess_answer_count = 0

    def update_statistics(self, answer, wv_suggestion):
        if answer == wv_suggestion:
            self.valid_answer_count += 1
        if wv_suggestion is not None:
            self.non_guess_answer_count += 1

    def select_initialized_model(self):
        wv = KeyedVectors.load(self.MODEL_NAME)
        if wv: return wv

        wv = api.load(self.MODEL_NAME)
        wv.save(self.MODEL_NAME)
        return wv

    def select_model_accuracy(self):
        return self.valid_answer_count / self.non_guess_answer_count

    def select_model_analysis_row(self):
        return self.name, self.vocabulary_size, self.valid_answer_count, self.non_guess_answer_count, self.select_model_accuracy()
