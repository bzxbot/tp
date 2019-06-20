import click

@click.command()
@click.option("--training-file-path", help="Path of the file to used in the training of the model")
@click.option("--testing-file-path", help="Path of the file to used in the testing of the model")
def bug_zapper(training_file_path, testing_file_path):
    csv_data_reader = CsvDataReader()
    file_data = csv_data_reader.read_data(training_file_path)
    test_file_data = csv_data_reader.read_data(testing_file_path)
    model_factory_creator = ModelFactoryCreator()
    model_factory = MLPModelFactory()
    model = model_factory.create_model()
    model.train(file_data)
    results = model.test(test_file_data)
    print("Hello world!")

def bug_zapper_process_data():
    print("Hello world!")

def bug_zapper_train_model():
    print("Hello world!")

def bug_zapper_test_model():
    print("Hello world!")

if __name__ == '__main__'
    bug_zapper()