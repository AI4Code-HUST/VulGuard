from vulguard.models.BaseWraper import BaseWraper
import json, torch, os
import torch.nn as nn
from .model import DeepJITModel
from vulguard.utils.utils import open_jsonl
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from .dataset import CustomDataset
from torch.utils.data import DataLoader

def get_auc(ground_truth, predict):
    roc_auc = roc_auc_score(y_true=ground_truth, y_score=predict)
    precisions, recalls, _ = precision_recall_curve(y_true=ground_truth, probas_pred=predict)
    pr_auc = auc(recalls, precisions)
    
    return roc_auc, pr_auc

class Com(BaseWraper):
    def __init__(self, language, device="cpu", **kwarg):
        self.model_name = 'com'
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
        self.val_loader = None
                
        self.default_input = "patch"
        
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
            
            checkpoint = torch.load(f"{model_path}/com.pth")  # Load the last saved checkpoint
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.total_loss = checkpoint['loss']

        # Set initialized to True
        self.initialized = True

    def preprocess(self, data_df, **kwarg):  
        print(f"Load data: {data_df}")
        data = pd.read_json(data_df, orient="records", lines=True)   
        labels = data.loc[:, "label"] if "label" in data.columns else None      
                  
        data = CustomDataset(data, self.hyperparameters, self.code_dictionary, self.message_dictionary)
        data_loader = DataLoader(data, self.hyperparameters["batch_size"])
        return data_loader, labels
    
        
    def postprocess(self, commit_ids, outputs, threshold, labels=None, **kwargs):
        result = pd.DataFrame({
            "commit_id": commit_ids,
            "probability": outputs,
        })
        result["prediction"] = (result["probability"] > threshold).astype(float)
        
        if labels is not None:
            result["label"] = labels

        return result

    def inference(self, infer_df, threshold, **kwarg):       
        data_loader, labels = self.preprocess(infer_df) if self.val_loader is None else (self.val_loader, None)
        
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
        
        final_prediction = self.postprocess(commit_ids, predicts, threshold, labels)

        return final_prediction
    
    def train(self, train_df, val_df, **kwarg):
        params = kwarg.get("params")
        save_path = kwarg.get("save_path")   
        threshold = 0.5 if params.threshold is None else params.threshold  
        criterion = nn.BCELoss()
        if self.optimizer is None:   
            self.optimizer = torch.optim.Adam(self.get_parameters(), lr=self.hyperparameters["learning_rate"]) 
        
        data_loader, train_labels = self.preprocess(train_df)
        assert train_labels is not None, "Ensure there is label column in training data"
        
        self.val_loader, val_ground_truth = self.preprocess(val_df)
        assert val_ground_truth is not None, "Ensure there is label column in validation data"

        
        best_valid_score = 0
        early_stop_count = 5
        self.last_epoch = self.hyperparameters["epoch"] if params.epochs is None else params.epochs
        for epoch in range(self.start_epoch, self.last_epoch + 1):
            print(f'Training: Epoch {epoch} / {self.last_epoch} -- Start')
            for batch in tqdm(data_loader):
                # Extract data from DataLoader
                code = batch["code"].to(self.device)
                message = batch["message"].to(self.device)
                label = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                predict = self.model(message, code)

                loss = criterion(predict, label)
                loss.backward()
                self.total_loss = loss.item()
                self.optimizer.step()

            print(f'Training: Epoch {epoch} / {self.last_epoch} -- Total loss: {self.total_loss}')

            prediction = self.inference(val_df, threshold)
            val_predict = prediction.loc[:, "probability"]
            
            roc_auc, pr_auc = get_auc(val_ground_truth, val_predict)
            print('Valid data -- ROC-AUC score:', roc_auc,  ' -- PR-AUC score:', pr_auc)

            valid_score = pr_auc
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                print('Save a better model', best_valid_score.item())
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
            
    
    def save(self, save_path, epoch=None, **kwarg):
        os.makedirs(save_path, exist_ok=True)
        
        save_path = f"{save_path}/com.pth"
        torch.save({
            'epoch': self.last_epoch if epoch is None else epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.total_loss,
        }, save_path)
    
