from src.selectors_x import select_synonym_dataset, select_best_similarity_from_model
from src.word_vector_model.word_vector_model import WordVectorModel

wv = WordVectorModel().model
for row in select_synonym_dataset()[0:5]:
    question, answer, value_0, value_1, value_2, value_3 = row
    wv_suggestion = select_best_similarity_from_model(question,
                                                      [value_0, value_1, value_2, value_3], wv)
    print("human_answer: " + answer + " model suggestion: " + wv_suggestion)
