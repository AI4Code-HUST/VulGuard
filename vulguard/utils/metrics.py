import pandas as pd
import numpy as np
import json
import math
import os
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, auc   
)

def eval_metrics(result_df, model, columns):
    pred = result_df['prediction']
    y_test = result_df['label']
    y_proba = result_df["probability"]
    
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

    if "LOC" not in result_df.columns:
        metric_df = pd.DataFrame([[roc_auc, pr_auc, acc, f1, prc, rc, mcc]], 
                             columns=columns, index=[model])
        return metric_df

    # find Effort metrics
    
    result_df['defect_density'] = result_df['proba'] / result_df['LOC']  # predicted defect density
    result_df['actual_defect_density'] = result_df['label'] / result_df['LOC']  # defect density

    result_df = result_df.sort_values(by='defect_density', ascending=False)
    actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
    actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)

    result_df['cum_LOC'] = result_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()
    real_buggy_commits = result_df[result_df['label'] == 1]
    
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
    
    metric_df = pd.DataFrame([[roc_auc, pr_auc, acc, f1, prc, rc, mcc, effort_at_20_percent_LOC_recall, recall_20_percent_effort, p_opt]], 
                             columns=columns, index=[model])
    return metric_df

def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort / 100) * result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
    recall_k_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    return recall_k_percent_effort

def get_metrics(predict_df, model, features_file=None):
    if features_file is not None:
        columns = ["roc_auc", "pr_auc", "accuracy", "f1_score", "precision", "recall", "mcc", "Effort@20", "Recall@20", "Popt"]
    else:
        columns = ["roc_auc", "pr_auc", "accuracy", "f1_score", "precision", "recall", "mcc"]
              
    predict_df.columns = ["commit_id", "probability", "prediction", "label"]
    predict_df['prediction'] = predict_df['prediction'].apply(lambda x: float(bool(x)))
    
    if features_file is not None:
        features_df = pd.read_json(features_file, lines=True)
        LOC_df = features_df[["commit_id", "la", "ld"]]
        LOC_df["LOC"] = LOC_df["la"] + LOC_df["ld"]
        LOC_df = LOC_df[["commit_id", "LOC"]]
        
        predict_df = pd.merge(predict_df, LOC_df, how="inner", on="commit_id")
        
    return eval_metrics(predict_df, model, columns)           

            