import csv
import os

from src.selectors_x import select_details_csv_rows


def delete_file_if_present(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)


def output_pre_trained_synonyms_test_details(synonym_test_details):
    delete_file_if_present("model-details.csv")
    writer = csv.writer(open("model-details.csv", 'a+'), delimiter=",")
    csv_rows = select_details_csv_rows(synonym_test_details)
    writer.writerows(csv_rows)

