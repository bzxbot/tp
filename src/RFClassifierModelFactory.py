from sklearn.ensemble import RandomForestClassifier
from RFClassifierModel import RFClassifierModel

class RFClassifierModelFactory:
    def create_model(self):
        sc_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        return RFClassifierModel(sc_model)