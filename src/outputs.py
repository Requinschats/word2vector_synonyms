import csv
import os


def delete_file_if_present(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)


def output_pre_trained_synonyms_test_details():
    delete_file_if_present("model-details.csv")
    file = open("model-details.csv", 'a+')
    csv.writer(file, delimiter=",")
    test_row = (1, "A towel,", 1.0),

