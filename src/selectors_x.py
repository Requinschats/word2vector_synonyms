import csv


def select_word_embeddings_similarity(word_1, word_2, wv):
    return wv.similarity(word_1, word_2)


def select_synonym_dataset():
    csv_reader = csv.reader(open("inputs/synonyms.csv"), delimiter=",")
    return [row for row in csv_reader]


def select_word_in_vocabulary(word, model):
    return word in list(model.index_to_key)


def select_is_one_option_in_vocabulary(options, model):
    for option in options:
        if select_word_in_vocabulary(option, model): return True
    return False


def select_guess_suggestion_from_model(question, options, model):
    is_question_in_vocabulary = select_word_in_vocabulary(question, model)
    is_one_option_in_vocabulary = select_is_one_option_in_vocabulary(options, model)
    return not is_question_in_vocabulary or not is_one_option_in_vocabulary


def select_best_similarity_from_model(question, options, model):
    best_similarity_score = 0
    best_similarity_option = ""
    for option in options:
        if not select_word_in_vocabulary(option, model): continue
        option_score = model.similarity(question, option)
        if option_score > best_similarity_score:
            best_similarity_score = option_score
            best_similarity_option = option
    return best_similarity_option


def select_details_csv_rows(synonym_test_details):
    csv_rows = []
    for test_detail in synonym_test_details:
        row, wv_suggestion = test_detail
        question, answer, value_0, value_1, value_2, value_3 = row
        label = select_synonym_question_label(answer, wv_suggestion)
        csv_row = (question, answer, wv_suggestion, label)
        csv_rows.append(csv_row)
    return csv_rows


def select_synonym_question_label(answer, wv_suggestion):
    if wv_suggestion is None:
        return "guess"
    if answer == wv_suggestion:
        return "correct"
    else:
        return "wrong"


def select_synonyms_test_details(word_vector):
    wv = word_vector.model
    synonym_test_details = []
    for row in select_synonym_dataset():
        question, answer, value_0, value_1, value_2, value_3 = row
        options = [value_0, value_1, value_2, value_3]

        wv_suggestion = None
        if not select_guess_suggestion_from_model(question, options, wv):
            wv_suggestion = select_best_similarity_from_model(question, options, wv)

        synonym_test_details.append((row, wv_suggestion))
        word_vector.update_statistics(answer, wv_suggestion)
    return synonym_test_details
