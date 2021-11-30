from src.outputs import output_pre_trained_synonyms_test_details, output_model_analysis, \
    delete_output_files_if_present, plot_accuracy_graph, plot_number_of_guesses_graph
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

accuracy_statistics = [[], []]
guesses_statistics = [[], []]
for model in model_list:
    print("Generating statistics for model " + model.name)
    synonym_test_details = select_synonyms_test_details(model)

    output_pre_trained_synonyms_test_details(model, synonym_test_details)
    output_model_analysis(model)

    accuracy_statistics[0].append(model.name)
    guesses_statistics[0].append(model.name)
    accuracy_statistics[1].append(model.select_model_accuracy())
    guesses_statistics[1].append(80 - model.non_guess_answer_count)

plot_accuracy_graph(accuracy_statistics)
plot_number_of_guesses_graph(guesses_statistics)
