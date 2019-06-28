from sklearn.neural_network import MLPClassifier
from MLPClassifierModel import MLPClassifierModel

class MLPClassifierModelFactory:
    def create_model(self):
        sc_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,), random_state=1)
        return MLPClassifierModel(sc_model)