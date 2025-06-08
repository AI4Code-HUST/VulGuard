from importlib.resources import files
import os
import json

SRC_PATH = str(files('vulguard'))

def sort_by_predict(commit_list):
    # Sort the list of dictionaries based on the "predict" value in descending order
    sorted_list = sorted(commit_list, key=lambda x: x['probability'], reverse=True)
    return sorted_list

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

def create_dg_cache(save_folder):
    dg_cache_path = f"{save_folder}/dg_cache"
    folders = ["save", "repo", "dataset"]
    if not os.path.exists(dg_cache_path):
        os.mkdir(dg_cache_path)
    for folder in folders:
        if not os.path.exists(os.path.join(dg_cache_path, folder)):
            os.mkdir(os.path.join(dg_cache_path, folder))
    return dg_cache_path
