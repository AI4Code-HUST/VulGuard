import os, torch, pickle
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import torch.nn as nn
from .models import (
    DeepJIT,
    SimCom,
    LAPredict,
    LogisticRegression,
    TLELModel as TLEL,
)
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from .utils.padding import padding_data, padding_data_point
from .utils.utils import yield_jsonl, open_jsonl
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
import pandas as pd

def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)
    return lr_auc

def init_model(model_name, language, device):   
    if  model_name == "deepjit":
        return DeepJIT(language=language, device=device)
    elif  model_name == "simcom":
        return SimCom(language=language, device=device)
    elif  model_name == "lapredict":
        return LAPredict(language=language)
    elif  model_name == "tlel":
        return TLEL(language=language)
    elif  model_name == "lr":
        return LogisticRegression(language=language)
    else:
        raise Exception("No such model")

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        commit_hash = self.data[0][idx]
        code = torch.tensor(self.data[1][idx])
        message = torch.tensor(self.data[2][idx])
        label = torch.tensor(self.data[3][idx], dtype=torch.float32)
        return {
            "commit_hash": commit_hash,
            "code": code,
            'message': message,
            "labels": label
        }
        
def load_dataset(file_path, hyperparameters, code_dict, msg_dict):
    commit_hashes, codes, messages, labels = [], [], [], []
    with tqdm(desc="Load Dataset: ") as bar:
        for data_point in yield_jsonl(file_path):
            commit_hashes.append(data_point["commit_id"])
            codes.append(padding_data_point(data_point=data_point["code_change"].split("\n"), dictionary=code_dict, params=hyperparameters, type='code'))
            messages.append(padding_data_point(data_point=data_point["messages"], dictionary=msg_dict, params=hyperparameters, type='msg'))
            labels.append(data_point["label"])
            bar.update(1)
    commit_hashes = np.array(commit_hashes)
    codes = np.array(codes)
    messages = np.array(messages)
    labels = np.array(labels)
    return (commit_hashes, codes, messages, labels)
    
def training_deep_learning(model, params, dg_cache_path):
    commit_path = f'{dg_cache_path}/dataset/{params.repo_name}/commit'
    dictionary_path = f'{commit_path}/dict.jsonl' if params.dictionary is None else params.dictionary
    train_set_path = f'{commit_path}/{params.model}_{params.repo_name}_train.jsonl' if params.commit_train_set is None else params.commit_train_set
    val_set_path = f'{commit_path}/{params.model}_{params.repo_name}_val.jsonl' if params.commit_val_set is None else params.commit_val_set
    if params.model == "simcom":
        model_save_path = f'{dg_cache_path}/save/{params.repo_name}/com.pth'
    else:
        model_save_path = f'{dg_cache_path}/save/{params.repo_name}/{params.model}.pth'
        
    if not os.path.exists(f'{dg_cache_path}/save/{params.repo_name}'):
        os.mkdir(f'{dg_cache_path}/save/{params.repo_name}')

    print("Create path!")
    # Init model
    if params.from_pretrain:
        model.initialize()
    else:
        model.initialize(dictionary=dictionary_path)
    print("Init model!")
    # Load dataset
    train_data = load_dataset(train_set_path, model.hyperparameters, model.code_dictionary, model.message_dictionary)

    if params.model == "simcom":
        val_data = load_dataset(val_set_path, model.hyperparameters, model.code_dictionary, model.message_dictionary)

    code_dataset = CustomDataset(train_data)
    code_dataloader = DataLoader(code_dataset, batch_size=model.hyperparameters['batch_size'])
    del train_data

    if params.model == "simcom":
        val_code_dataset = CustomDataset(val_data)
        val_code_dataloader = DataLoader(val_code_dataset, batch_size=model.hyperparameters['batch_size'])
        del val_data
    
    optimizer = torch.optim.Adam(model.get_parameters(), lr=params.learning_rate)
    criterion = nn.BCELoss()

    # Validate
    best_valid_score = 0
    smallest_loss = 1000000
    early_stop_count = 5
    start_epoch = 1
    total_loss = 0

    if params.from_pretrain:
        checkpoint = torch.load(model_save_path)  # Load the last saved checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        total_loss = checkpoint['loss']

    for epoch in range(start_epoch, params.epochs + 1):
        print(f'Training: Epoch {epoch} / {params.epochs} -- Start')
        for batch in tqdm(code_dataloader):
            # Extract data from DataLoader
            code = batch["code"].to(model.device)
            message = batch["message"].to(model.device)
            labels = batch["labels"].to(model.device)

            optimizer.zero_grad()

            # ---------------------- DefectGuard -------------------------------
            predict = model(message, code)
            # ------------------------------------------------------------------
            
            loss = criterion(predict, labels)
            loss.backward()
            total_loss = loss.item()
            optimizer.step()

        print(f'Training: Epoch {epoch} / {params.epochs} -- Total loss: {total_loss}')

        if params.model == "simcom":
            model.com.eval()
            with torch.no_grad():
                all_predict, all_label = [], []
                for batch in tqdm(val_code_dataloader):
                    # Extract data from DataLoader
                    code = batch["code"].to(params.device)
                    message = batch["message"].to(params.device)
                    labels = batch["labels"].to(params.device)

                    # Forward
                    predict = model(message, code)
                    all_predict += predict.cpu().detach().numpy().tolist()
                    all_label += labels.cpu().detach().numpy().tolist()

            auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
            auc_pc_score = auc_pc(all_label, all_predict)
            print('Valid data -- ROC-AUC score:', auc_score,  ' -- PC-AUC score:', auc_pc_score)

            valid_score = auc_pc_score
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                print('Save a better model', best_valid_score.item())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.com.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, model_save_path)
            else:
                print('No update of models', early_stop_count)
                if epoch > 5:
                    early_stop_count = early_stop_count - 1
                if early_stop_count < 0:
                    break
        else:
            print(total_loss < smallest_loss, total_loss, smallest_loss)
            if total_loss < smallest_loss:
                smallest_loss = total_loss
                print('Save a better model', smallest_loss)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, model_save_path)
            else:
                print('No update of models', early_stop_count)
                if epoch > 5:
                    early_stop_count = early_stop_count - 1
                if early_stop_count < 0:
                    break

def training_machine_learning(model, params, dg_cache_path):
    train_df_path = f'{dg_cache_path}/dataset/{params.repo_name}/feature/{params.repo_name}_train.jsonl' if params.feature_train_set is None else params.feature_train_set
    train_df = pd.read_json(train_df_path, lines=True)

    cols = (
        ["la"]
        if model.model_name == "lapredict"
        else [
            "ns",
            "nd",
            "nf",
            "entropy",
            "la",
            "ld",
            "lt",
            "fix",
            "ndev",
            "age",
            "nuc",
            "exp",
            "rexp",
            "sexp",
        ]
    )
    X_train = train_df.loc[:, cols]
    y_train = train_df.loc[:, "label"]
    
    if not os.path.exists(f'{dg_cache_path}/save/{params.repo_name}'):
        os.mkdir(f'{dg_cache_path}/save/{params.repo_name}')

    if model.model_name == "simcom":
        X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
        model.sim.fit(X_train, y_train)
        model.save_sim(f'{dg_cache_path}/save/{params.repo_name}')
    elif model.model_name == "lapredict" or model.model_name == "lr":
        model.model = sk_LogisticRegression(class_weight='balanced', max_iter=1000)
        model.model.fit(X_train, y_train)
        model.save(f'{dg_cache_path}/save/{params.repo_name}')
    elif model.model_name == "tlel":
        model.model.fit(X_train, y_train)
        model.save(f'{dg_cache_path}/save/{params.repo_name}')
        pass
    else:
        raise Exception("No such model")

def training(params):
    # create save folders
    dg_cache_path = f"{params.dg_save_folder}/dg_cache"
    folders = ["save", "repo", "dataset"]
    if not os.path.exists(dg_cache_path):
        os.mkdir(dg_cache_path)
    for folder in folders:
        if not os.path.exists(os.path.join(dg_cache_path, folder)):
            os.mkdir(os.path.join(dg_cache_path, folder))

    model = init_model(params.model, params.repo_language, params.device)
    if params.model in ["deepjit", "simcom"]:
        training_deep_learning(model, params, dg_cache_path)

    if params.model in ["lapredict", "lr", "tlel", "simcom"]:
        training_machine_learning(model, params, dg_cache_path)

    

    