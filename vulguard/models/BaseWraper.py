from abc import ABC, abstractmethod

class BaseWraper(ABC):

    @abstractmethod
    def initialize(self,  **kwarg):
        pass

    @abstractmethod
    def preprocess(self, **kwarg):
        pass

    @abstractmethod
    def postprocess(self, **kwarg):
        pass

    @abstractmethod
    def train(self, **kwarg):
        pass
    
    @abstractmethod
    def inference(self, **kwarg):
        pass
    
    @abstractmethod
    def save(self, **kwarg):
        pass