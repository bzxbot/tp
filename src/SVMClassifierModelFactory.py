from sklearn import svm
from SVMClassifierModel import SVMClassifierModel

class SVMClassifierModelFactory:
    def create_model(self):
        sc_model = svm.SVC(gamma='scale')
        return SVMClassifierModel(sc_model)