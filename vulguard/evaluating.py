import os
from .models.init_model import init_model
from .utils.utils import SRC_PATH, create_dg_cache
from .utils.metrics import get_metrics
import pandas as pd
          
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
    test_df_path = f'{dg_cache_path}/dataset/{params.repo_name}/data/test_{model.default_input}_{params.repo_name}.jsonl' if params.test_set is None else params.test_set
    test_df = pd.read_json(test_df_path, orient="records", lines=True)
    label_df = test_df[["commit_id", "label"]]
    
    result_df = model.inference(infer_df=test_df_path, threshold=threshold, params=params)
    result_df = result_df.merge(label_df, on="commit_id", how="inner")
    result_df.to_csv(f'{predict_score_path}/{model.model_name}.csv', index=False, columns=["commit_id", "label", "prediction", "probability"])
    print(f"Predict scores saved to: {predict_score_path}/{model.model_name}.csv")
    
    size_file = params.size_file if params.size_file is not None else None
    metrics_df = get_metrics(result_df, model.model_name, size_file)    
    metrics_df.to_csv(f'{result_path}/{model.model_name}.csv', index=True)
    print(f"Metrics saved to: {result_path}/{model.model_name}.csv")