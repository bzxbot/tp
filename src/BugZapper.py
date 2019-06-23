import click
import pickle
from CsvDataReader import CsvDataReader
from DengueDataScrubber import DengueDataScrubber
from MLPClassifierModelFactory import MLPClassifierModelFactory
from SVMClassifierModelFactory import SVMClassifierModelFactory
from ModelSerializer import ModelSerializer

@click.command()
@click.option("--train-file", help="Path of the file to used in the training of the model")
@click.option("--test-file", help="Path of the file to used in the testing of the model")
@click.option("-s/-ns", "--save/--no-save", help="Serialize the model")
@click.option("-l/-nl", "--load/--no-load", help="Loads an existing model")
def bug_zapper(train_file, test_file, save, load):
    csv_data_reader = CsvDataReader()
    file_data = csv_data_reader.read_data(train_file, ";")
    test_file_data = csv_data_reader.read_data(test_file, ";")
    data_scrubber = DengueDataScrubber()
    X_train, y_train = data_scrubber.scrub_data(file_data)
    X_test, y_test = data_scrubber.scrub_data(test_file_data)
    mlp_model_factory = MLPClassifierModelFactory()
    svm_model_factory = SVMClassifierModelFactory()
    svm_model = svm_model_factory.create_model()
    svm_model.train(X_train, y_train)
    mlp_model = mlp_model_factory.create_model()
    model_serializer = ModelSerializer()
    if load:
        model_serializer.load_model(mlp_model)
    else:
        mlp_model.train(X_train, y_train)
    if save:
        model_serializer.save_model(mlp_model)
    mlp_results = mlp_model.predict(X_test, y_test)
    svm_resuls = svm_model.predict(X_test, y_test)
    print(mlp_results)
    print(svm_resuls)

def bug_zapper_process_data():
    print("Hello world!")

def bug_zapper_train_model():
    print("Hello world!")

def bug_zapper_test_model():
    print("Hello world!")

if __name__ == '__main__':
    bug_zapper()