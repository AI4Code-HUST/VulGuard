from abc import ABC, abstractmethod

class BaseWraper(ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def preprocess(self, data):
        pass

    @abstractmethod
    def postprocess(self, commit_ids, inference_output, threshold):
        pass

    @abstractmethod
    def train(self, train_df, val_df):
        pass
    
    @abstractmethod
    def inference(self, infer_df, threshold):
        pass
    
    @abstractmethod
    def save(self, save_dir):
        pass