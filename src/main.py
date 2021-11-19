from src.outputs import output_pre_trained_synonyms_test_details, output_model_analysis, \
    delete_output_files_if_present
from src.selectors_x import select_synonyms_test_details
from src.word_vector_model.word_vector_model import WordVectorModel

delete_output_files_if_present()

word_vector_google_news_300 = WordVectorModel()
word_vector_glove_wiki_gigaword_200 = WordVectorModel("glove-wiki-gigaword-200")
word_vector_glove_wiki_gigaword_300 = WordVectorModel("glove-wiki-gigaword-300")
word_vector_glove_twitter_25 = WordVectorModel("glove-twitter-25")
word_vector_glove_twitter_200 = WordVectorModel("glove-twitter-200")

model_list = [word_vector_google_news_300, word_vector_glove_wiki_gigaword_200,
              word_vector_glove_wiki_gigaword_300, word_vector_glove_twitter_25,
              word_vector_glove_twitter_200]

# for model in model_list:
# synonym_test_details = select_synonyms_test_details(word_vector_google_news_300)
# output_pre_trained_synonyms_test_details(synonym_test_details)
# output_model_analysis(word_vector_google_news_300)
