from defectguard.models.BaseWraper import BaseWraper
import pickle, json, torch, os
from .model import DeepJITModel
from defectguard.utils.utils import download_folder, SRC_PATH, open_jsonl
from .utils import *

class DeepJIT(BaseWraper):
    def __init__(self, language='cpp', device="cpu"):
        self.model_name = 'deepjit'
        self.language = language
        self.initialized = False
        self.model = None
        self.device = device
        self.message_dictionary = None
        self.code_dictionary = None
        self.hyperparameters = None
        # download_folder(self.model_name, self.language)

    def __call__(self, message, code):
        return self.model(message, code)
    
    def get_parameters(self):
        return self.model.parameters()
    
    def set_device(self, device):
        self.device = device
    
    def initialize(self, dictionary=None, hyperparameters=None, from_pretrain=True, state_dict=None):
        # Load dictionary
        if dictionary:
            dictionary = open_jsonl(dictionary)
        else:
            dictionary = open_jsonl(f"{SRC_PATH}/models/metadata/{self.model_name}/{self.language}_dictionary.jsonl")
        self.message_dictionary, self.code_dictionary = dictionary[0], dictionary[1]

        # Load parameters
        if hyperparameters:
            with open(hyperparameters, 'r') as file:
                self.hyperparameters = json.load(file)
        else:
            with open(f"{SRC_PATH}/models/hyperparameters.json", 'r') as file:
                self.hyperparameters = json.load(file)

        # Set up param
        self.hyperparameters["filter_sizes"] = [int(k) for k in self.hyperparameters["filter_sizes"].split(',')]
        self.hyperparameters["vocab_msg"], self.hyperparameters["vocab_code"] = len(self.message_dictionary), len(self.code_dictionary)
        self.hyperparameters["class_num"] = 1

        # Create model and Load pretrain
        self.model = DeepJITModel(self.hyperparameters).to(device=self.device)
        if from_pretrain and dictionary is None:
            checkpoint = torch.load(f"{SRC_PATH}/models/metadata/{self.model_name}/{self.language}", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif state_dict:
            checkpoint = torch.load(state_dict, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Set initialized to True
        self.initialized = True

    def preprocess(self, data):
        pass

    def inference(self, model_input):
        if not self.initialized:
            self.initialize()

        # Forward
        self.model.eval()
        with torch.no_grad():
            # Extract data from DataLoader
            code = torch.tensor(model_input["code"], device=self.device)
            message = torch.tensor(model_input["message"], device=self.device)

            # Forward
            predict = self.model(message, code)
        
        return predict

    def postprocess(self, og_commit_hashes, commit_hashes, inference_output):
        if not self.initialized:
            self.initialize()

        inference_output = inference_output.tolist()

        result = []
        for i in range(len(commit_hashes)):
            if commit_hashes[i] == 'Not code change':
                result.append({'commit_hash': og_commit_hashes[i], 'predict': -1})
            else:
                result.append({'commit_hash': commit_hashes[i], 'predict': inference_output[i]})

        return result

    def handle(self, data):
        if not self.initialized:
            self.initialize()
            
        model_output = self.inference(data)
        final_prediction = self.postprocess(data['commit_hashes'], data, model_output)

        return final_prediction
    
    def save(self, save_dir):
        if not os.path.isdir(save_dir):       
            os.makedirs(save_dir)
        
        save_path = f"{save_dir}/deepjit.pth"
        torch.save(self.model.state_dict(), save_path)
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)