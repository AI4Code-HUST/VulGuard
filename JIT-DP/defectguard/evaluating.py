import os, torch, pickle
from torch.utils.data import Dataset, DataLoader
from .models import (
    DeepJIT,
    SimCom,
    LAPredict,
    LogisticRegression,
    TLELModel as TLEL,
)
from .utils.padding import padding_data, padding_data_point
from .utils.logger import logger, logs
from .utils.utils import yield_jsonl, open_jsonl
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    matthews_corrcoef, 
    precision_recall_curve, 
    auc
)
from datetime import datetime

def init_model(model_name, language, device):
    if model_name == "deepjit":
        return DeepJIT(language=language, device=device)
    elif model_name == "simcom":
        return SimCom(language=language, device=device)
    elif model_name == "lapredict":
        return LAPredict(language=language)
    elif model_name == "tlel":
        return TLEL(language=language)
    elif model_name == "lr":
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
        label = torch.tensor(self.data[3][idx], dtype=torch.float32)
        code = torch.tensor(self.data[1][idx])
        message = torch.tensor(self.data[2][idx])
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

def evaluating_deep_learning(pretrain, params, dg_cache_path):
    commit_path = f'{dg_cache_path}/dataset/{params.repo_name}/commit'
    dictionary_path = f'{commit_path}/{params.repo_name}_train_dict.pkl' if params.dictionary is None else params.dictionary
    test_set_path = f'{commit_path}/{params.model}_{params.repo_name}_test.pkl' if params.commit_test_set is None else params.commit_test_set
    pretrain_path = f'{dg_cache_path}/save/{params.repo_name}/{pretrain}'
    
    if not os.path.exists(f'{dg_cache_path}/save/{params.repo_name}'):
        os.mkdir(f'{dg_cache_path}/save/{params.repo_name}')

    # Init model
    model = init_model(params.model, params.repo_language, params.device)

    if params.from_pretrain:
        model.initialize()
    else:
        model.initialize(dictionary=dictionary_path, state_dict=pretrain_path)

    # Load dataset
    test_data = load_dataset(test_set_path, model.hyperparameters, model.code_dictionary, model.message_dictionary)
    code_dataset = CustomDataset(test_data)
    code_dataloader = DataLoader(code_dataset, batch_size=1)

    if model.model_name == "simcom":
        model.com.eval()    
    else:
        model.model.eval()
    with torch.no_grad():
        commit_hashes, all_predict, all_label = [], [], []
        for batch in tqdm(code_dataloader):
            # Extract data from DataLoader
            commit_hashes.append(batch['commit_hash'][0])
            code = batch["code"].to(params.device)
            message = batch["message"].to(params.device)
            labels = batch["labels"].to(params.device)

            # Forward
            predict = model(message, code)
            all_predict += predict.cpu().detach().numpy().tolist()
            all_label += labels.cpu().detach().numpy().tolist()
            
            # Free GPU memory
            del code, message, labels, predict
            torch.cuda.empty_cache()

    return commit_hashes, all_predict, all_label

def evaluating_machine_learning(pretrain, params, dg_cache_path):
    test_df_path = f'{dg_cache_path}/dataset/{params.repo_name}/feature/{params.repo_name}_test.csv' if params.feature_test_set is None else params.feature_test_set
    test_df = pd.read_json(test_df_path, lines=True)
    model = init_model(params.model, params.repo_language, params.device)
    
    if not os.path.exists(f'{dg_cache_path}/save/{params.repo_name}'):
        os.mkdir(f'{dg_cache_path}/save/{params.repo_name}')


    if params.from_pretrain:
        model.initialize()
    elif model.model_name == "simcom":
        model.initialize(pretrain=f'{dg_cache_path}/save/{params.repo_name}/{pretrain}', dictionary=params.dictionary)
    else:
        model.initialize(pretrain=f'{dg_cache_path}/save/{params.repo_name}/{pretrain}')

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
    commit_hashes = test_df.loc[:, "commit_id"].to_list()
    X_test = test_df.loc[:, cols]
    y_test = test_df.loc[:, "label"]

    y_proba = model.predict_proba(X_test)[:, 1]

    return commit_hashes, y_proba, y_test
        
def get_pretrain(model_name):
    if model_name == "deepjit":
        return "deepjit.pth"
    elif model_name == "sim":
        return "sim.pkl"
    elif model_name == "com":
        return "com.pth"
    elif model_name == "lapredict":
        return "lapredict.pkl"
    elif model_name == "lr":
        return "lr.pkl"
    elif model_name == "tlel":
        return "tlel.pkl"
    else:
        raise Exception("No such model")
    
def average(proba_1, proba_2):
    if len(proba_1) != len(proba_2):
        raise ValueError("Both lists must be of the same length")
    return [(x + y) / 2 for x, y in zip(proba_1, proba_2)]

def metrics(ground_truth, probability, threshold=0.5):
    roc_auc = roc_auc_score(y_true=ground_truth,  y_score=probability)
    precision, recall, _ = precision_recall_curve(y_true=ground_truth, probas_pred=probability)
    pr_auc = auc(recall, precision)
    
    predict = [1 if proba > threshold else 0 for proba in probability]
    f1 = f1_score(y_true=ground_truth, y_pred=predict)
    mcc = matthews_corrcoef(y_true=ground_truth, y_pred=predict)
        
    return predict, roc_auc, pr_auc, f1, mcc

def evaluating(params):
    # create save folders
    dg_cache_path = f"{params.dg_save_folder}/dg_cache"
    folders = ["save", "repo", "dataset"]
    predict_score_path = f'{dg_cache_path}/save/{params.repo_name}/predict_scores/'
    resutl_path = f'{dg_cache_path}/save/{params.repo_name}/results/'

    if not os.path.exists(dg_cache_path):
        os.mkdir(dg_cache_path)
    for folder in folders:
        if not os.path.exists(os.path.join(dg_cache_path, folder)):
            os.mkdir(os.path.join(dg_cache_path, folder))
    if os.path.isdir(predict_score_path) is False:
        os.makedirs(predict_score_path)
    if os.path.isdir(resutl_path) is False:
        os.makedirs(resutl_path)

    if params.model in ["deepjit", "simcom"]:
        model_name = params.model if params.model != "simcom" else "com"
        pretrain = get_pretrain(model_name)
        com_hashes, com_proba, ground_truth = evaluating_deep_learning(pretrain, params, dg_cache_path)
        com_preds, com_roc_auc, com_pr_auc, com_f1, com_mcc = metrics(ground_truth, com_proba)

        logs(f'{dg_cache_path}/save/{params.repo_name}/results/roc_auc.csv', params.repo_name, com_roc_auc, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/pr_auc.csv', params.repo_name, com_pr_auc, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/f1.csv', params.repo_name, com_f1, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/mcc.csv', params.repo_name, com_mcc, model_name)
        
        df = pd.DataFrame({'commit_hash': com_hashes, 'label': ground_truth, 'proba': com_proba, 'pred': com_preds})
        df.to_csv(f'{dg_cache_path}/save/{params.repo_name}/predict_scores/{model_name}.csv', index=False, sep=',')

    if params.model in ["lapredict", "lr", "tlel", "simcom"]:
        model_name = params.model if params.model != "simcom" else "sim"
        pretrain = get_pretrain(model_name)
        sim_hashes, sim_proba, ground_truth = evaluating_machine_learning(pretrain, params, dg_cache_path)
        sim_preds, sim_roc_auc, sim_pr_auc, sim_f1, sim_mcc = metrics(ground_truth, sim_proba)

        logs(f'{dg_cache_path}/save/{params.repo_name}/results/roc_auc.csv', params.repo_name, sim_roc_auc, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/pr_auc.csv', params.repo_name, sim_pr_auc, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/f1.csv', params.repo_name, sim_f1, model_name)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/mcc.csv', params.repo_name, sim_mcc, model_name)
        
        df = pd.DataFrame({'commit_hash': sim_hashes, 'label': ground_truth, 'proba': sim_proba, 'pred': sim_preds})
        df.to_csv(f'{dg_cache_path}/save/{params.repo_name}/predict_scores/{model_name}.csv', index=False, sep=',')
    
    if params.model in ["simcom"]:
        assert sorted(com_hashes) == sorted(sim_hashes)
        model_name = "simcom"
        simcom_proba = average(sim_proba, com_proba)
        preds, roc_auc, pr_auc, f1, mcc = metrics(ground_truth, simcom_proba)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/roc_auc.csv', params.repo_name, roc_auc, params.model)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/pr_auc.csv', params.repo_name, pr_auc, params.model)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/f1.csv', params.repo_name, f1, params.model)
        logs(f'{dg_cache_path}/save/{params.repo_name}/results/mcc.csv', params.repo_name, mcc, params.model)
      
        df = pd.DataFrame({'commit_hash': sim_hashes, 'label': ground_truth, 'proba': simcom_proba, 'pred': preds})
        df.to_csv(f'{dg_cache_path}/save/{params.repo_name}/predict_scores/{model_name}.csv', index=False, sep=',')