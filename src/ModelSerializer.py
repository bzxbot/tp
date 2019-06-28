import pickle
import os

class ModelSerializer:
    model_dir = 'models'
    file_extension = '.pkl'
    
    def save_model(self, model):
        with open(os.path.join(self.model_dir, model.__class__.__name__ + self.file_extension), 'wb') as file_handle:
            pickle.dump(model, file_handle)
        
    def load_model(self, model):
        with open(os.path.join(self.model_dir, model.__class__.__name__ + self.file_extension), 'rb') as file_handle:
            return pickle.load(file_handle)