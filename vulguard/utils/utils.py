from importlib.resources import files
import os, gdown
import json
import math

SRC_PATH = str(files('vulguard'))

def sort_by_predict(commit_list):
    # Sort the list of dictionaries based on the "predict" value in descending order
    sorted_list = sorted(commit_list, key=lambda x: x['probability'], reverse=True)
    return sorted_list

def vsc_output(data):
    # Extract the commit hashes from "no_code_change_commit"
    no_code_change_commits = data.get("no_code_change_commit", [])
    
    # Extract the "deepjit" list
    deepjit_list = data.get("deepjit", [])
    
    # Create a dictionary with keys from "no_code_change_commit" and values as -1
    new_dict = [{'commit_id': commit, 'predict': -1} for commit in no_code_change_commits]
    
    # Append the new dictionary to the "deepjit" list
    deepjit_list += (new_dict)
    
    # Update the "deepjit" key in the original data
    data["deepjit"] = deepjit_list

    return data

def check_threshold(data, threshold):
    output = {}
    for model, predicts in data.items():
        if model == "no_code_change_commit":
            continue
        else:
            output[model] = []
        for predict in predicts:
            if predict['predict'] >= threshold:
                output[model].append(predict)
    
    return output

def yield_jsonl(file):  
    # Read the file and yield lines in equal parts
    with open(file, 'r') as f:
        for line in f:
            yield json.loads(line)

def open_jsonl(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data