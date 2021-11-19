import csv
import os

from src.selectors_x import select_details_csv_rows


def delete_file_if_present(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)


def output_pre_trained_synonyms_test_details(model, synonym_test_details):
    file_path = "outputs/" + model.name + "-details.csv"
    writer = csv.writer(open(file_path, 'a+'), delimiter=",")
    csv_rows = select_details_csv_rows(synonym_test_details)
    writer.writerows(csv_rows)


def output_model_analysis(word_vector):
    file_path = "outputs/analysis.csv"
    writer = csv.writer(open(file_path, 'a+'), delimiter=",")
    writer.writerows([word_vector.select_model_analysis_row()])


def delete_output_files_if_present():
    file_path = "outputs/model-details.csv"
    delete_file_if_present(file_path)
    file_path = "outputs/analysis.csv"
    delete_file_if_present(file_path)
