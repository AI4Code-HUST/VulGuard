import os, torch, pickle
from .utils.utils import SRC_PATH
from .models import (
    DeepJIT,
    SimCom,
    LAPredict,
    LogisticRegression,
    TLELModel as TLEL,
)
from tqdm import tqdm
import pandas as pd

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
    
def training(params):
    # create save folders
    dg_cache_path = f"{params.dg_save_folder}/dg_cache"
    folders = ["save", "repo", "dataset"]
    if not os.path.exists(dg_cache_path):
        os.mkdir(dg_cache_path)
    for folder in folders:
        if not os.path.exists(os.path.join(dg_cache_path, folder)):
            os.mkdir(os.path.join(dg_cache_path, folder))
            
    save_path = f'{dg_cache_path}/save/{params.repo_name}'
    model = init_model(params.model, params.repo_language, params.device)
    
    train_df_path = f'{dg_cache_path}/dataset/{params.repo_name}/data/train_{model.default_input}_{params.repo_name}.jsonl' if params.train_set is None else params.train_set
    val_df_path = f'{dg_cache_path}/dataset/{params.repo_name}/data/val_{model.default_input}_{params.repo_name}.jsonl' if params.val_set is None else params.val_set
    dictionary = f'{dg_cache_path}/dataset/{params.repo_name}/dict_{params.repo_name}.jsonl'  if params.dictionary is None else params.dictionary 
    hyperparameters = f"{SRC_PATH}/models/{model.model_name}/hyperparameters.json" if params.hyperparameters is None else params.hyperparameters
    
    if params.model_path is not None:
        model.initialize(model_path=params.model_path, dictionary=dictionary, hyperparameters=hyperparameters)
    
    print(f"Load train data: {train_df_path}")
    train_df = pd.read_json(train_df_path, orient="records", lines=True)
        
    print(f"Load validate data: {val_df_path}")
    val_df = pd.read_json(val_df_path, orient="records", lines=True)
    
    
    print(f"Train {model.model_name}")
    model.model = model.train(train_df, val_df, params=params, save_path=save_path, dictionary=dictionary)
    
    print(f"Save {model.model_name}")
    model.save(save_path)
    

    