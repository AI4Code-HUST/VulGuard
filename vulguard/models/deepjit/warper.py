from vulguard.models.BaseWraper import BaseWraper
import json, torch, os
import torch.nn as nn
from .model import DeepJITModel
from .dataset import CustomDataset, get_data_loader
from vulguard.utils.utils import open_jsonl
from tqdm import tqdm
import pandas as pd


class DeepJIT(BaseWraper):
    def __init__(self, language, device="cpu", **kwarg):
        self.model_name = 'deepjit'
        self.language = language
        self.initialized = False
        self.model = None
        self.device = device
        self.message_dictionary = None
        self.code_dictionary = None
        self.hyperparameters = None 
        self.optimizer = None
        self.start_epoch = 1
        self.last_epoch = 0 
        self.total_loss = 0  
        
        self.default_input = "merge"

    def __call__(self, message, code):
        return self.model(message, code)
    
    def get_parameters(self):
        return self.model.parameters()
    
    def set_device(self, device):
        self.device = device
    
    def initialize(self, dictionary, hyperparameters, model_path=None, **kwarg):
        # Load dictionary
        dictionary = open_jsonl(dictionary)
        self.message_dictionary, self.code_dictionary = dictionary[0], dictionary[1]
        
        # Load hyperparameter
        with open(hyperparameters, 'r') as file:
            self.hyperparameters = json.load(file)
            
        self.hyperparameters["filter_sizes"] = [int(k) for k in self.hyperparameters["filter_sizes"].split(',')]
        self.hyperparameters["vocab_msg"], self.hyperparameters["vocab_code"] = len(self.message_dictionary), len(self.code_dictionary)
        self.hyperparameters["class_num"] = 1

        
        if model_path is None:
            self.model = DeepJITModel(self.hyperparameters).to(device=self.device)
            
        else:        
            self.model = DeepJITModel(self.hyperparameters).to(device=self.device)
            self.optimizer = torch.optim.Adam(self.get_parameters())
            
            if model_path is not None:
                checkpoint = torch.load(f"{model_path}/deepjit.pth")  # Load the last saved checkpoint
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.total_loss = checkpoint['loss']

        # Set initialized to True
        self.initialized = True
    
    def preprocess(self, data_df, **kwarg):  
        print(f"Load data: {data_df}")
        data = pd.read_json(data_df, orient="records", lines=True)         
                  
        data = CustomDataset(data, self.hyperparameters, self.code_dictionary, self.message_dictionary)
        data_loader = get_data_loader(data, self.hyperparameters["batch_size"])
        return data_loader
    
        
    def postprocess(self, commit_ids, outputs, threshold, **kwarg):
        result = []
        for commit_id, output in zip(commit_ids, outputs):
            json_obj = {
                'commit_id': commit_id, 
                'probability': output, 
                'prediction': float(output > threshold)
            }
            result.append(json_obj)
        result = pd.DataFrame(result)
        return result

    def inference(self, infer_df, threshold, **kwarg):       
        data_loader = self.preprocess(infer_df)
        
        self.model.eval()
        with torch.no_grad():
            commit_ids, predicts = [], []
            for batch in tqdm(data_loader):
                # Extract data from DataLoader
                commit_ids.append(batch['commit_id'][0])
                code = batch["code"].to(self.device)
                message = batch["message"].to(self.device)

                # Forward
                predict = self.model(message, code)
                predicts += predict.cpu().detach().numpy().tolist()
                
                # Free GPU memory
                torch.cuda.empty_cache()
        
        final_prediction = self.postprocess(commit_ids, predicts, threshold)

        return final_prediction
    
    def train(self, train_df, **kwarg):
        params = kwarg.get("params")
        save_path = kwarg.get("save_path")        
        self.optimizer = torch.optim.Adam(self.get_parameters(), lr=params.learning_rate) if self.optimizer is None else self.optimizer
        criterion = nn.BCELoss()
        
        data_loader = self.preprocess(train_df)
        
        smallest_loss = 1000000
        early_stop_count = 5
        
        for epoch in range(self.start_epoch, params.epochs + 1):
            self.last_epoch = epoch
            print(f'Training: Epoch {epoch} / {params.epochs} -- Start')
            for batch in tqdm(data_loader):
                # Extract data from DataLoader
                code = batch["code"].to(self.device)
                message = batch["message"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                predict = self.model(message, code)

                loss = criterion(predict, labels)
                loss.backward()
                self.total_loss = loss.item()
                self.optimizer.step()

            print(f'Training: Epoch {epoch} / {params.epochs} -- Total loss: {self.total_loss}')

            print(self.total_loss < smallest_loss, self.total_loss, smallest_loss)
            if self.total_loss < smallest_loss:
                smallest_loss = self.total_loss
                print('Save a better model', smallest_loss)
                self.save(
                    save_path=save_path,
                    epoch=epoch,
                    optimizer=self.optimizer.state_dict(), 
                    loss=loss.item()
                )
                
            else:
                print('No update of models', early_stop_count)
                if epoch > 5:
                    early_stop_count = early_stop_count - 1
                if early_stop_count < 0:
                    break
            
    
    def save(self, save_path, **kwarg):
        os.makedirs(save_path, exist_ok=True)
        
        save_path = f"{save_path}/deepjit.pth"
        torch.save({
            'epoch': self.last_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.total_loss,
        }, save_path)
    
