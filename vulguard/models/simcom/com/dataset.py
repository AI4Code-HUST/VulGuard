import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .padding import padding_data_point

class CustomDataset(Dataset):
    def __init__(self, file_path, hyperparameters, code_dict, msg_dict):
        self.data = load_dataset(file_path, hyperparameters, code_dict, msg_dict)
        
    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        commit_id = self.data[0][idx]
        code = torch.tensor(self.data[1][idx])
        message = torch.tensor(self.data[2][idx])
        label = torch.tensor(self.data[3][idx], dtype=torch.float32) if self.data[3][idx] is not None else None
        
        item = {
            "commit_id": commit_id,
            "code": code,
            "message": message,
            "label": label
        }
        
        return item

        
def load_dataset(data_df, hyperparameters, code_dict, msg_dict):
    commit_ids = np.array(data_df["commit_id"])
    
    codes = np.array (
        [padding_data_point(data_point=code.split("\n"), dictionary=code_dict, params=hyperparameters, type='code')
        for code in list(data_df["code_change"])]
    )
    
    messages = np.array(
        [padding_data_point(data_point=mes, dictionary=msg_dict, params=hyperparameters, type='msg')
        for mes in list(data_df["messages"])]
    )
        
    if "label" in data_df.columns:
        labels = np.array(data_df["label"]) 
    else:
        labels = [None for i in range(len(commit_ids))]  
    
    return (commit_ids, codes, messages, labels)

def get_data_loader(data, batch_size):
    return DataLoader(data, batch_size)