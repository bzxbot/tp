import pandas as pd

class CsvDataReader:
    def read_data(self, file_path, separator):
        return pd.read_csv(file_path, sep=separator) 