import csv
import os
import matplotlib.pyplot as plt
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


def plot_accuracy_graph(statistics):
    x_values = statistics[0]
    x_values.append("Random")
    y_values = statistics[1]
    y_values.append(0.25)
    plt.rcParams["figure.figsize"] = (15, 10)
    plt.plot(x_values, y_values)
    plt.xlabel('Corpus name')
    plt.ylabel('Accuracy')
    plt.show()


def plot_number_of_guesses_graph(guesses_statistics):
    x_values = guesses_statistics[0]
    y_values = guesses_statistics[1]
    plt.rcParams["figure.figsize"] = (15, 10)
    plt.plot(x_values, y_values)
    plt.xlabel('Corpus name')
    plt.ylabel('Guess count')
    plt.show()


def create_directory_if_not_present(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
