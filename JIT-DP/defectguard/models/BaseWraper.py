from abc import ABC, abstractmethod

class BaseWraper(ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def preprocess(self, data):
        pass

    @abstractmethod
    def inference(self, model_input):
        pass

    @abstractmethod
    def postprocess(self, inference_output):
        pass

    @abstractmethod
    def handle(self, data):
        pass