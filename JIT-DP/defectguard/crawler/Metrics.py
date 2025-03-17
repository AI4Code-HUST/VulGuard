import pandas as pd
import numpy as np
import json
import math
import os
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, auc   
)

def eval_result(predict_file, features_file):
    predict_df = pd.read_csv(predict_file)
    # predict_df.columns = ["commit_hash", "proba", "pred", "label"]
    predict_df['pred'] = predict_df['pred'].apply(lambda x: float(bool(x)))
    features_df = pd.read_json(features_file, lines=True)
    
    LOC_df = features_df[["commit_id", "la", "ld"]]
    LOC_df["LOC"] = LOC_df["la"] + LOC_df["ld"]
    LOC_df = LOC_df[["commit_id", "LOC"]]
    del features_df
    
    merge_df = pd.merge(predict_df, LOC_df, how="inner", left_on="commit_hash", right_on="commit_id")
    merge_df = merge_df.drop("commit_id", axis=1)
    print(len(merge_df))
    return eval_metrics(merge_df)

def eval_metrics(result_df):
    pred = result_df['pred']
    y_test = result_df['label']
    y_proba = result_df["proba"]

    result_df['defect_density'] = result_df['proba'] / result_df['LOC']  # predicted defect density
    result_df['actual_defect_density'] = result_df['label'] / result_df['LOC']  # defect density

    result_df = result_df.sort_values(by='defect_density', ascending=False)
    actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
    actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)

    result_df['cum_LOC'] = result_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

    real_buggy_commits = result_df[result_df['label'] == 1]
    label_list = list(result_df['label'])
    all_rows = len(label_list)
    
    # find AUC
    roc_auc = roc_auc_score(y_true=y_test, y_score=y_proba)
    precisions, recalls, _ = precision_recall_curve(y_true=y_test, probas_pred=y_proba)
    pr_auc = auc(recalls, precisions)

    # find metrics
    acc = accuracy_score(y_true=y_test, y_pred=pred)
    f1 = f1_score(y_true=y_test, y_pred=pred)
    prc = precision_score(y_true=y_test, y_pred=pred)
    rc = recall_score(y_true=y_test, y_pred=pred)
    mcc = matthews_corrcoef(y_true=y_test, y_pred=pred)
    
    # find Recall@20%Effort
    cum_LOC_20_percent = 0.2 * result_df.iloc[-1]['cum_LOC']
    buggy_line_20_percent = result_df[result_df['cum_LOC'] <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label'] == 1]
    recall_20_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    # find Effort@20%Recall
    buggy_20_percent = real_buggy_commits.head(math.ceil(0.2 * len(real_buggy_commits)))
    buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
    effort_at_20_percent_LOC_recall = int(buggy_20_percent_LOC) / float(result_df.iloc[-1]['cum_LOC'])

    # find P_opt
    percent_effort_list = []
    predicted_recall_at_percent_effort_list = []
    actual_recall_at_percent_effort_list = []
    actual_worst_recall_at_percent_effort_list = []

    for percent_effort in np.arange(10, 101, 10):
        predicted_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, result_df, real_buggy_commits)
        actual_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_result_df, real_buggy_commits)
        actual_worst_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_worst_result_df, real_buggy_commits)

        percent_effort_list.append(percent_effort / 100)

        predicted_recall_at_percent_effort_list.append(predicted_recall_k_percent_effort)
        actual_recall_at_percent_effort_list.append(actual_recall_k_percent_effort)
        actual_worst_recall_at_percent_effort_list.append(actual_worst_recall_k_percent_effort)

    p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                 (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))
    
    result_df = pd.DataFrame([[roc_auc, pr_auc, acc, f1, prc, rc, mcc, effort_at_20_percent_LOC_recall, recall_20_percent_effort, p_opt]], 
                             columns=["roc_auc", "pr_auc", "accuracy", "f1_score", "precision", "recall", "mcc", "Effort@20", "Recall@20", "Popt"])
    return result_df

def merge(path):
    sim_df = pd.read_csv(f"{path}/sim.csv")
    com_df = pd.read_csv(f"{path}/com.csv")

    merged_df = pd.merge(sim_df, com_df, on=['commit_hash', 'label'], how='outer', suffixes=('_df1', '_df2'))

    # Calculate the average score
    merged_df['proba'] = merged_df[['proba_df1', 'proba_df2']].mean(axis=1)

    # Select only the required columns
    final_df = merged_df[['commit_hash', 'label', 'proba']]
    y_test = final_df.loc[:, "label"]
    y_proba = final_df.loc[:, "proba"]
    y_pred = [1 if i >= 0.5 else 0 for i in y_proba]
    final_df["pred"] = y_pred
    
    final_df.to_csv(f'{path}/simcom.csv', index=False)

def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort / 100) * result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
    recall_k_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    return recall_k_percent_effort

def run(params):
    predict_file = params.predict_file
    features_file = params.features_file
    save_folder = params.save_folder
    model = params.model
    
    combine_df = pd.DataFrame([], columns=["roc_auc", "pr_auc", "accuracy", "f1_score", "precision", "recall", "mcc", "Effort@20", "Recall@20", "Popt"])
    result_df = eval_result(predict_file, features_file)
    result_df.index = [model]
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    if os.path.exists(save_folder + "/metrics.csv"):
        combine_df = pd.read_csv(save_folder + "/metrics.csv", index_col=0)
        
    combine_df = pd.concat([combine_df, result_df], axis=0)
    combine_df.to_csv(save_folder + "/metrics.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file", type=str, required=True)
    parser.add_argument("--features_file", type=str, default=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    
    params = parser.parse_args()
    run(params)
    
    

# if __name__ == "__main__":
#     for setup in ["SETUP1", "SETUP2", "SETUP3"]:
#         for sampling in ["ros", "rus", "unsampling"]:
#             combine_df = pd.DataFrame([], columns=["roc_auc", "pr_auc", "accuracy", "f1_score", "precision", "recall", "mcc", "Effort@20", "Recall@20", "Popt"])
#             for model in ["jitfine"]:
#                 if model == "simcom":
#                     merge(f"E:/Containers/{setup}/{sampling}/dg_cache/save/FFmpeg/predict_scores")
                
#                 print(f"{setup} - {sampling} - {model}: ")
#                 path = f"E:\Containers\{setup}\{sampling}\dg_cache\save\FFmpeg\{model}\checkpoints"
#                 predict_file = f"{path}\predictions.csv"
#                 features_file = f"E:/JIT-VP-Data/FFmpeg/{setup}/{setup}-FFmpeg-features-test.jsonl"
#                 result_df = eval_result(predict_file, features_file)
#                 result_df.index = [model]
                
#                 if not os.path.exists(f"E:/JIT-DP-experiment/save/{sampling}/{setup}"):
#                     os.makedirs(f"E:/JIT-DP-experiment/save/{sampling}/{setup}")
#                 combine_df = pd.concat([combine_df, result_df], axis=0)
#                 combine_df.to_csv(f"E:/JIT-DP-experiment/save/{sampling}/{setup}/{setup}_{sampling}.csv")
            