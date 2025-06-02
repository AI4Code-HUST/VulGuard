import os, torch, pickle
from .models import (
    DeepJIT,
    SimCom,
    LAPredict,
    LogisticRegression,
    TLELModel as TLEL,
)
from .utils.utils import SRC_PATH
from .utils.metrics import get_metrics
import pandas as pd

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
    
def evaluating(params):
    dg_cache_path = f"{params.dg_save_folder}/dg_cache"
    folders = ["save", "repo", "dataset"]
    if not os.path.exists(dg_cache_path):
        os.mkdir(dg_cache_path)
    for folder in folders:
        if not os.path.exists(os.path.join(dg_cache_path, folder)):
            os.mkdir(os.path.join(dg_cache_path, folder))
        
    predict_score_path = f'{dg_cache_path}/save/{params.repo_name}/predict_scores/'
    result_path = f'{dg_cache_path}/save/{params.repo_name}/results/'
    os.makedirs(predict_score_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    
    # Init model
    model = init_model(params.model, params.repo_language, params.device)    
    dictionary = f'{dg_cache_path}/dataset/{params.repo_name}/dict_{params.repo_name}.jsonl'  if params.dictionary is None else params.dictionary 
    hyperparameters = f"{SRC_PATH}/models/{model.model_name}/hyperparameters.json" if params.hyperparameters is None else params.hyperparameters
    
    if params.model_path is not None:
        model.initialize(model_path=params.model_path, dictionary=dictionary, hyperparameters=hyperparameters)
    else:
        model.initialize(model_path=f'{dg_cache_path}/save/{params.repo_name}', dictionary=dictionary, hyperparameters=hyperparameters)
    
    threshold = params.threshold if params.threshold is not None else 0.5
    test_df_path = f'{dg_cache_path}/dataset/{params.repo_name}/data/test_{model.default_input}_{params.repo_name}.jsonl' if params.test_set is None else params.test_set
    test_df = pd.read_json(test_df_path, orient="records", lines=True)
    
    result_df = model.inference(test_df, threshold)
    label_df = test_df[["commit_id", "label"]]
    result_df = result_df.merge(label_df, on="commit_id", how="inner")
    result_df.to_csv(f'{predict_score_path}/{model.model_name}.csv', index=False, columns=["commit_id", "label", "prediction", "probability"])
    
    size_file = params.size_file if params.size_file is not None else None
    metrics_df = get_metrics(result_df, model.model_name, size_file)    
    metrics_df.to_csv(f'{result_path}/{model.model_name}.csv', index=True)