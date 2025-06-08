from . import *
from .utils.logger import logger
from .utils.utils import SRC_PATH, sort_by_predict, create_dg_cache
import os, json, time
import pandas as pd

def inferencing(params):
    logger("Start DefectGuard")
    dg_cache_path = create_dg_cache(params.dg_save_folder)        
    infer_path = f'{dg_cache_path}/save/{params.repo_name}/infer/'
    os.makedirs(infer_path, exist_ok=True)
    
    # Init model
    start_whole_process = time.time()
    model = init_model(params.model, params.repo_language, params.device)    
    
    dictionary = f'{dg_cache_path}/dataset/{params.repo_name}/dict_{params.repo_name}.jsonl'  if params.dictionary is None else params.dictionary 
    hyperparameters = f"{SRC_PATH}/models/{model.model_name}/hyperparameters.json" if params.hyperparameters is None else params.hyperparameters
    model_path = f'{dg_cache_path}/save/{params.repo_name}' if params.model_path is None else params.model_path
    print(f"Init model: {model.model_name}")
    model.initialize(model_path=model_path, dictionary=dictionary, hyperparameters=hyperparameters)

    threshold = params.threshold if params.threshold is not None else 0.5
    infer_df_path = params.infer_set
    infer_df = pd.read_json(infer_df_path, orient="records", lines=True)
    result_df = model.inference(infer_df=infer_df, threshold=threshold)
    end_whole_process = time.time()

    if not params.no_warning:
        if params.ranking:
            rank_df = result_df.drop("prediction", axis=1, errors='ignore')
            rank_records = rank_df.to_dict(orient="records")
            rank_records = sort_by_predict(rank_records)
            print("Ranking of the most dangerous commits:")
            print(json.dumps(rank_records, indent=2))
            
        if params.predict:
            predict_df = result_df.drop("probability", axis=1, errors='ignore')
            predict_records = predict_df.to_dict(orient="records")
            print(f"Prediction with threshold {params.threshold}:")
            print(json.dumps(predict_records, indent=2))

        if  not params.predict and not params.ranking:
            result_records = result_df.to_dict(orient="records")
            print("Inference results:")
            print(json.dumps(result_records, indent=2))
        
    result_df.to_json(f"{infer_path}/infer_result.jsonl", orient="records", lines=True)    
    print(f"Inference time: {end_whole_process - start_whole_process}")
    print(f"Result save to: {infer_path}/infer_result.jsonl")

