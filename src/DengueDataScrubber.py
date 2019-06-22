class DengueDataScrubber:
    y_column = 'tp_classificacao_final'
    
    def scrub_data(self, data):
        X = data.drop(columns=[self.y_column]).select_dtypes(exclude=['object']).fillna(0)
        y = data[self.y_column] 
        return X, y
