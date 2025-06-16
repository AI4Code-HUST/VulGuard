from .models.init_model import init_model
from .utils.utils import SRC_PATH, create_dg_cache
    
def training(params):
    # create save folders
    dg_cache_path = create_dg_cache(params.dg_save_folder)
    save_path = f'{dg_cache_path}/save/{params.repo_name}'
    model = init_model(params.model, params.repo_language, params.device)
    
    default_inputs = model.default_input.split(",")
    if params.train_set:
        train_df_path = params.train_set
    else:
        train_df_path = ','.join([f'{dg_cache_path}/dataset/{params.repo_name}/data/train_{default_input}_{params.repo_name}.jsonl' for default_input in default_inputs])
    
    if params.val_set:
        val_df_path = params.val_set
    else:
        val_df_path = ','.join([f'{dg_cache_path}/dataset/{params.repo_name}/data/val_{default_input}_{params.repo_name}.jsonl' for default_input in default_inputs])
        
    model_path = params.model_path
    dictionary = f'{dg_cache_path}/dataset/{params.repo_name}/dict_{params.repo_name}.jsonl'  if params.dictionary is None else params.dictionary 
    hyperparameters = f"{SRC_PATH}/models/{model.model_name}/hyperparameters.json" if params.hyperparameters is None else params.hyperparameters
    
    print(f"Init model: {model.model_name}")
    model.initialize(model_path=model_path, dictionary=dictionary, hyperparameters=hyperparameters)
    
    print(f"Train {model.model_name}")
    save_best_path = f"{save_path}/models/best_epoch"
    model.train(train_df=train_df_path, val_df=val_df_path, params=params, save_path=save_best_path)
    
    print(f"Save {model.model_name}")
    save_last_path = f"{save_path}/models/last_epoch"
    model.save(save_path=save_last_path)
    print(f"Model saved to: {save_path}")
    

    