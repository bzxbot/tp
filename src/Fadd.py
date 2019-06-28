import click
import pickle
from sklearn.model_selection import train_test_split
from CsvDataReader import CsvDataReader
from DataScrubber import DataScrubber
from MLPClassifierModelFactory import MLPClassifierModelFactory
from SVMClassifierModelFactory import SVMClassifierModelFactory
from RFClassifierModelFactory import RFClassifierModelFactory
from ModelSerializer import ModelSerializer
from FeatureScore import FeatureScore
from ResultsMetrics import ResultsMetrics

@click.command()
@click.option("--file", help="Path of the file to used in the training of the model")
@click.option("-s/-ns", "--save/--no-save", help="Serialize the model")
@click.option("-l/-nl", "--load/--no-load", help="Loads an existing model")
@click.option("--feature-score", "--load/--no-load", help="Loads an existing model")
def fadd(file, save, load, feature_score):
    csv_data_reader = CsvDataReader()
    file_data = csv_data_reader.read_data(file, ";")
    train, test = train_test_split(file_data, test_size=0.3)
    data_scrubber = DataScrubber()
    X_train, y_train = data_scrubber.scrub_data(train)
    X_test, y_test = data_scrubber.scrub_data(test)
    mlp_model_factory = MLPClassifierModelFactory()
    svm_model_factory = SVMClassifierModelFactory()
    rf_model_factory = RFClassifierModelFactory()
    svm_model = svm_model_factory.create_model()
    mlp_model = mlp_model_factory.create_model()
    rf_model = rf_model_factory.create_model()
    model_serializer = ModelSerializer()
    if load:
        mlp_model = model_serializer.load_model(mlp_model)
        rf_model = model_serializer.load_model(rf_model)
        svm_model = model_serializer.load_model(svm_model)
    else:
        mlp_model.train(X_train, y_train)
        rf_model.train(X_train, y_train)
        svm_model.train(X_train, y_train)
    if save:
        model_serializer.save_model(mlp_model)
        model_serializer.save_model(svm_model)
        model_serializer.save_model(rf_model)
    if feature_score:
        feature_score = FeatureScore()
        feature_score.evaluate_features(X_train, y_train)
    mlp_y_pred = mlp_model.predict(X_test)
    svm_y_pred = svm_model.predict(X_test)
    rf_y_pred = rf_model.predict(X_test)
    result_metrics = ResultsMetrics()
    result_metrics.print_result_metrics(y_test, mlp_y_pred, svm_y_pred, rf_y_pred)

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    fadd()