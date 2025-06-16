import os
from .models.init_model import init_model
from .utils.utils import SRC_PATH, create_dg_cache
from .utils.metrics import get_metrics
import pandas as pd
import numpy as np
          
def evaluating(params):
    dg_cache_path = create_dg_cache(params.dg_save_folder)
    predict_score_path = f'{dg_cache_path}/save/{params.repo_name}/predict_scores'
    result_path = f'{dg_cache_path}/save/{params.repo_name}/results'
    os.makedirs(predict_score_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    
    # Init model
    model = init_model(params.model, params.repo_language, params.device)    
    dictionary = f'{dg_cache_path}/dataset/{params.repo_name}/dict_{params.repo_name}.jsonl'  if params.dictionary is None else params.dictionary 
    hyperparameters = f"{SRC_PATH}/models/{model.model_name}/hyperparameters.json" if params.hyperparameters is None else params.hyperparameters
    model_path = f'{dg_cache_path}/save/{params.repo_name}/models/best_epoch' if params.model_path is None else params.model_path
    print(f"Init model: {model.model_name}")
    model.initialize(model_path=model_path, dictionary=dictionary, hyperparameters=hyperparameters)
    
    threshold = 0.5 if params.threshold is None else params.threshold 
    default_inputs = model.default_input.split(",")
    if params.test_set:
        test_df_path = params.test_set
    else:
        test_df_path = ','.join([f'{dg_cache_path}/dataset/{params.repo_name}/data/test_{default_input}_{params.repo_name}.jsonl' for default_input in default_inputs]) 
    
    result_df = model.inference(infer_df=test_df_path, threshold=threshold, params=params)
    result_df.to_csv(f'{predict_score_path}/{model.model_name}.csv', index=False, columns=["commit_id", "label", "prediction", "probability"])
    print(f"Predict scores saved to: {predict_score_path}/{model.model_name}.csv")
    
    size_df_path = f'{dg_cache_path}/dataset/{params.repo_name}/data/test_Kamei_features_{params.repo_name}.jsonl' if params.size_set is None else params.size_set
    size_df = pd.read_json(size_df_path, orient="records", lines=True)
    if not (
        'commit_id' in size_df.columns and
        'commit_id' in result_df.columns and
        np.array_equal(size_df["commit_id"].values, result_df["commit_id"].values)
    ):
        size_df_path = None
    
    metrics_df = get_metrics(result_df, model.model_name, size_df_path)    
    metrics_df.to_csv(f'{result_path}/{model.model_name}.csv', index=True)
    print(f"Metrics saved to: {result_path}/{model.model_name}.csv")