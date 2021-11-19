from gensim.models import KeyedVectors
import gensim.downloader as api


class WordVectorModel:
    DEFAULT_MODEL_NAME = 'word2vec-google-news-300'

    def __init__(self, model_name=DEFAULT_MODEL_NAME):
        self.name = model_name
        self.model = self.select_initialized_model()
        self.vocabulary_size = len(list(self.model.index_to_key))
        self.valid_answer_count = 0
        self.non_guess_answer_count = 0

    def update_statistics(self, answer, wv_suggestion):
        if answer == wv_suggestion:
            self.valid_answer_count += 1
        if wv_suggestion is not None:
            self.non_guess_answer_count += 1

    def select_initialized_model(self):
        wv = None
        try:
            wv = KeyedVectors.load("saved_models/" + self.name)
        except:
            wv = api.load(self.name)
            wv.save(self.name)
        return wv

    def select_model_accuracy(self):
        return self.valid_answer_count / self.non_guess_answer_count

    def select_model_analysis_row(self):
        return self.name, self.vocabulary_size, self.valid_answer_count, self.non_guess_answer_count, self.select_model_accuracy()
