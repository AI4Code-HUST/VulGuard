from . import *
from .utils.utils import SRC_PATH, create_dg_cache
import pandas as pd
    
def training(params):
    # create save folders
    dg_cache_path = create_dg_cache(params.dg_save_folder)
    save_path = f'{dg_cache_path}/save/{params.repo_name}'
    model = init_model(params.model, params.repo_language, params.device)
    
    train_df_path = f'{dg_cache_path}/dataset/{params.repo_name}/data/train_{model.default_input}_{params.repo_name}.jsonl' if params.train_set is None else params.train_set
    val_df_path = f'{dg_cache_path}/dataset/{params.repo_name}/data/val_{model.default_input}_{params.repo_name}.jsonl' if params.val_set is None else params.val_set
    dictionary = f'{dg_cache_path}/dataset/{params.repo_name}/dict_{params.repo_name}.jsonl'  if params.dictionary is None else params.dictionary 
    hyperparameters = f"{SRC_PATH}/models/{model.model_name}/hyperparameters.json" if params.hyperparameters is None else params.hyperparameters
    model_path = None if params.model_path is None else params.model_path
    
    print(f"Init model: {model.model_name}")
    model.initialize(model_path=model_path, dictionary=dictionary, hyperparameters=hyperparameters)
    
    print(f"Load train data: {train_df_path}")
    train_df = pd.read_json(train_df_path, orient="records", lines=True)
        
    print(f"Load validate data: {val_df_path}")
    val_df = pd.read_json(val_df_path, orient="records", lines=True)
    
    
    print(f"Train {model.model_name}")
    model.train(train_df=train_df, val_df=val_df, params=params, save_path=save_path)
    
    print(f"Save {model.model_name}")
    model.save(save_path=save_path)
    print(f"Model saved to: {save_path}")
    

    