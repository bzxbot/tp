from sklearn.neural_network import MLPClassifier
from MLModel import MLModel

class MLPClassiferModel(MLModel):
    def __init__(self, model):
        self.model = model

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, y):
        return self.model.predict(X.values)