from sklearn.metrics import accuracy_score, f1_score, recall_score

class ResultsMetrics:
    def print_result_metrics(self, y_test, mlp_y_pred, svm_y_pred, rf_y_pred):
        print("MLP Accuracy:", accuracy_score(y_test, mlp_y_pred))
        print("SVM Accuracy: ", accuracy_score(y_test, svm_y_pred))
        print("RF Accuracy", accuracy_score(y_test, rf_y_pred))
        print("MLP F1:", f1_score(y_test, mlp_y_pred, average='weighted'))
        print("SVM F1: ", f1_score(y_test, svm_y_pred, average='weighted'))
        print("RF F1", f1_score(y_test, rf_y_pred, average='weighted'))
        print("MLP Recall:", recall_score(y_test, mlp_y_pred, average='weighted'))
        print("SVM Recall: ", recall_score(y_test, svm_y_pred, average='weighted'))
        print("RF Recall", recall_score(y_test, rf_y_pred, average='weighted'))