import csv


def select_word_embeddings_similarity(word_1, word_2, wv):
    return wv.similarity(word_1, word_2)


def select_synonym_dataset():
    csv_reader = csv.reader(open("inputs/synonyms.csv"), delimiter=",")
    return [row for row in csv_reader]


def select_best_similarity_from_model(question, options, model):
    best_similarity_score = 0
    best_similarity_option = ""
    for option in options:
        option_score = model.similarity(question, option)
        if option_score > best_similarity_score:
            best_similarity_score = option_score
            best_similarity_option = option
    return best_similarity_option
