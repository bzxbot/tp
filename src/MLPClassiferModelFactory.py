from sklearn.neural_network import MLPClassifier
from MLPClassiferModel import MLPClassiferModel

class MLPClassiferModelFactory:
    def create_model(self):
        sc_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        return MLPClassiferModel(sc_model)