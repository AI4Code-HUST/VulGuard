import json
import math
import re, os, shutil
import tempfile
import traceback
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import List, Set, Tuple
from tqdm import tqdm
from .utils.utils import *

log = create_console_log_handler("console")

def get_value(file: str, key: str) -> Set[str]:
    out = set()
    for line in load_jsonl(file):
        out.add(line.get(key, None))    
    return out

def get_VFC_VIC(label_files: List[str]) -> (Set[str], Set[str]):
    VFC, VIC = set(), set()
    for label_file in label_files:
        log.info(label_file)
        for line in load_jsonl(label_file):
            VFC.add(line["VFC"])
            VIC.update(line["VIC"])
    return (VFC, VIC)

def get_non_VIC_and_security(files: List[str], VFC: Set[str], VIC: Set[str]) -> Set[str]:
    non_VIC, security = set(), set()
    for file in files:
        for commit in tqdm(load_jsonl(file)):
            if commit["commit_id"] not in (VFC | VIC):
                non_VIC.add(commit["commit_id"])
            if commit["fix"] == 1:
                security.add(commit["commit_id"])
    return non_VIC, security

def assign_date(target_files: List[str], output_folder, VIC, VFC, non_VIC, security, non_sec_VFC, non_sec_non_VIC) -> None:
    for target_file in target_files:
        for line in load_jsonl(target_file):
            out = {
                "commit_id": line["commit_id"],
                "date": line["date"]
            }
            sec = 0
            if line["fix"] ==  1:
                with open(f"{output_folder}/UNSPLIT/security.jsonl", "a") as f:
                    f.write(json.dumps(out) + "\n")
                    sec = 1
            
            if line["commit_id"] in VIC:
                with open(f"{output_folder}/UNSPLIT/VIC.jsonl", "a") as f:
                    f.write(json.dumps(out) + "\n")
            
            elif line["commit_id"] in VFC:
                with open(f"{output_folder}/UNSPLIT/VFC.jsonl", "a") as f:
                    f.write(json.dumps(out) + "\n")
                    
                if sec == 0:
                    with open(f"{output_folder}/UNSPLIT/non_sec_VFC.jsonl", "a") as f:
                        f.write(json.dumps(out) + "\n")
            
            else:
                with open(f"{output_folder}/UNSPLIT/non_VIC.jsonl", "a") as f:
                    f.write(json.dumps(out) + "\n")
                
                if sec == 0:
                    with open(f"{output_folder}/UNSPLIT/non_sec_non_VIC.jsonl", "a") as f:
                        f.write(json.dumps(out) + "\n")          
            del out
                        
def get_data_sorted_by_date(path: str) -> List[Dict]:
    res = [v for v in load_jsonl(path)]
    res = sorted(res, key=lambda x: x["date"])
    save_jsonl(res, path)

def split_by_ratio(data: List, ratios: List) -> List[List]:
    out = []
    for r in ratios:
        start = math.ceil(r[0] * len(data))
        end = math.ceil(r[1] * len(data))
        out.append(data[start:end])
    return out

def split_by_date(data: List, date: List[Tuple[int, int]]) -> List[List]:
    out = [[] for i in range(len(date))]
    for d in data:
        for i, (start, end) in enumerate(date):
            if d["date"] >= start and d["date"] <= end:
                out[i].append(d)
    return out

def de_date(data: List[List[Dict]]) -> List[List[str]]:
    res = []
    for dtset in data:
        res.append([])
        for point in dtset:
            res[-1].append(point["commit_id"])
            
    return res

def to_file(lines: List, part: str, label0s: List, label1s: List, temp_dir: str, id: int) -> str:
    # log.info(f"{file} - {part}")
    temp_files = {
        dataset: {
            setup: tempfile.NamedTemporaryFile(mode='w', delete=False, dir=temp_dir, suffix=f"_{part}_{dataset}_{setup}.tmp")
            for setup in range(5)
        } for dataset in ["train", "val", "test"]
    }
    
    names = []
    for dataset in ["train", "val", "test"]:
        for setup in range(5):
            names.append(temp_files[dataset][setup].name)

    datasets = ["train", "val", "test"]
    count = 0
    for line in tqdm(lines, f"Chunk {id}"):
        for setup in range(5):
            for dataset, label0, label1 in zip(datasets, label0s[setup], label1s[setup]):
                if line["commit_id"] in label0:
                    line["label"] = 0
                    with open(temp_files[dataset][setup].name, "a") as f:
                        f.write(json.dumps(line) + "\n")

                
                elif line["commit_id"] in label1:
                    line["label"] = 1
                    with open(temp_files[dataset][setup].name, "a") as f:
                        f.write(json.dumps(line) + "\n")

    return names

def merge_class_files(temp_files: List, output_files: Dict, part: str) -> None:
    log.info("Merge")
    class_files = {
        part: {
            dataset: {
                setup: set() for setup in range(5)
            }
            for dataset in ["train", "val", "test"]
        } 
        for part in ["features", "simcom", "deepjit", "vcc-features"]    
    }

    for temp_file in temp_files:
        setup_name = int(temp_file.split('_')[-1].split('.')[0])
        dataset_name = temp_file.split('_')[-2]
        part_name = temp_file.split('_')[-3]
        class_files[part_name][dataset_name][setup_name].add(temp_file)
    
    for dataset in ["train", "val", "test"]:
        for setup in range(5):
            output_file = output_files[part][dataset][setup]
            with open(output_file, 'w') as f_out:
                for temp_file in class_files[part][dataset][setup]:
                    with open(temp_file, 'r') as f_in:
                        f_out.write(f_in.read())
                            
def read_file_in_chunks(filename, chunk_size=2000):
    with open(filename, 'r') as file:
        while True:
            lines = [file.readline() for _ in range(chunk_size)]
            lines = list(filter(None, lines))
            if not lines:
                break
            lines = [json.loads(line) for line in lines]
            yield lines
                            
def to_dataset(project: str, out_folder: str, label0s: List[List[List]], label1s: List[List[List]], workers: int=8) -> None:
    for setup in range(5):
        if not os.path.exists(f"{out_folder}/SETUP{setup+1}/unsampling"):
            os.makedirs(f"{out_folder}/SETUP{setup+1}/unsampling")
    
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    input_files = {
        "features": find_files(FEATURES_PATERN, f"{DEFAULT_EXTRACTED_OUTPUT}/{project}"),
        "simcom": find_files(SIMCOM_PATERN, f"{DEFAULT_EXTRACTED_OUTPUT}/{project}"),
        "deepjit": find_files(DEEPJIT_PATERN, f"{DEFAULT_EXTRACTED_OUTPUT}/{project}"),
        "vcc-features": find_files(VCCFINDER_PATERN, f"{DEFAULT_EXTRACTED_OUTPUT}/{project}")
    }
    
    output_files = { 
        part: {
                "train": {setup : f"{out_folder}/SETUP{setup+1}/unsampling/SETUP{setup+1}-{project}-{part}-train.jsonl" for setup in range(5)},
                "val": {setup : f"{out_folder}/SETUP{setup+1}/SETUP{setup+1}-{project}-{part}-val.jsonl" for setup in range(5)},
                "test": {setup : f"{out_folder}/SETUP{setup+1}/SETUP{setup+1}-{project}-{part}-test.jsonl" for setup in range(5)}
            } 
        for part in ["features", "simcom", "deepjit", "vcc-features"]
    }
    for part in ["vcc-features"]:
        file = input_files[part][0]
        temp_files = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            log.info(file)
            id = 0
            for chunk in tqdm(read_file_in_chunks(file), "Process Chunks: "):
                id += 1
                futures.append(executor.submit(to_file, chunk, part, label0s, label1s, temp_dir, id))
                    
            for future in as_completed(futures):
                temp_files.extend(future.result())

        # Merge the temp files for each class into final output files
        merge_class_files(temp_files, output_files, part)
    shutil.rmtree(temp_dir)                                   
     
def check_before_run(output_folder: str) -> bool:
    if not os.path.exists(f"{output_folder}/UNSPLIT/VIC.jsonl"):
        return False
    if not os.path.exists(f"{output_folder}/UNSPLIT/VFC.jsonl"):
        return False
    if not os.path.exists(f"{output_folder}/UNSPLIT/non_VIC.jsonl"):
        return False
    if not os.path.exists(f"{output_folder}/UNSPLIT/security.jsonl"):
        return False
    if not os.path.exists(f"{output_folder}/UNSPLIT/non_sec_VFC.jsonl"):
        return False
    if not os.path.exists(f"{output_folder}/UNSPLIT/non_sec_non_VIC.jsonl"):
        return False
    return True

def run(params):
    input_folder = params.input_folder
    output_folder = params.output_folder if params.output_folder else DEFAULT_DATA_OUTPUT
    project = params.project
    output_folder = f"{output_folder}/{project}"
    
    trusted_file = TRUSTED_LABEL_PATERN.format(project)
    semi_trusted_file = SEMI_TRUSTED_LABEL_PATERN.format(project)
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    if not params.continue_run or not check_before_run(output_folder):   
        if os.path.exists(f"{output_folder}/UNSPLIT"):
            shutil.rmtree(f"{output_folder}/UNSPLIT")
         
        T_VFC, T_VIC = get_VFC_VIC(find_files(trusted_file, f"{input_folder}/trusted_data/{project}"))
        log.info(find_files(trusted_file, f"{input_folder}/trusted_data/{project}"))
        ST_VFC, ST_VIC = get_VFC_VIC(find_files(semi_trusted_file, f"{input_folder}/semi_trusted_data/{project}"))
        log.info(find_files(semi_trusted_file, f"{input_folder}/semi_trusted_data/{project}"))
        VFC = T_VFC | ST_VFC
        VIC = T_VIC | ST_VIC
        del T_VFC, T_VIC, ST_VFC, ST_VIC
        log.info("Complete get VFC, VIC")
        feature_files = find_files(FEATURES_PATERN, f"{DEFAULT_EXTRACTED_OUTPUT}/{project}")
        log.info(feature_files)
        non_VIC, security = get_non_VIC_and_security(feature_files, VFC, VIC)
        log.info(non_VIC)
        non_sec_VFC = [elem for elem in VFC if elem not in security]
        non_sec_non_VIC = [elem for elem in non_VIC if elem not in security]
        
        log.info("Complete get nonVIC")
        if not os.path.exists(f"{output_folder}/UNSPLIT"):
            os.mkdir(f"{output_folder}/UNSPLIT")
        assign_date(feature_files, output_folder, VIC, VFC, non_VIC, security, non_sec_VFC, non_sec_non_VIC)
        
        log.info("Complete assign date!")
        
        get_data_sorted_by_date(f"{output_folder}/UNSPLIT/VIC.jsonl")
        get_data_sorted_by_date(f"{output_folder}/UNSPLIT/VFC.jsonl")
                
        log.info("Sorted by date")
        del VIC, VFC, non_VIC
        
    VIC = read_jsonl(f"{output_folder}/UNSPLIT/VIC.jsonl")
    VFC = read_jsonl(f"{output_folder}/UNSPLIT/VFC.jsonl")
    non_VIC = read_jsonl(f"{output_folder}/UNSPLIT/non_VIC.jsonl")
    security = read_jsonl(f"{output_folder}/UNSPLIT/security.jsonl")
    non_sec_VFC = read_jsonl(f"{output_folder}/UNSPLIT/non_sec_VFC.jsonl")
    non_sec_non_VIC = read_jsonl(f"{output_folder}/UNSPLIT/non_sec_non_VIC.jsonl")
    
    log.info(f"VFC: {len(VFC)}")
    log.info(f"VIC: {len(VIC)}")
    log.info(f"non_VIC: {len(non_VIC)}")
    log.info(f"security: {len(security)}")
    log.info(f"non_sec_VFC: {len(non_sec_VFC)}")
    log.info(f"non_sec_non_VIC: {len(non_sec_non_VIC)}")
    
    if params.unsplit_only:
        exit()
        
    ratios = [(0, 0.75), (0.75, 0.8), (0.8, 1)]
    VIC = split_by_ratio(VIC, ratios)
    date = [(v[0]["date"], v[-1]["date"]) for v in VIC]
    VFC = split_by_date(VFC, date)
    non_VIC = split_by_date(non_VIC, date)
    non_sec_VFC = split_by_date(non_sec_VFC, date)
    non_sec_non_VIC = split_by_date(non_sec_non_VIC, date)
    
    log.info("Splitted")
    VIC = de_date(VIC)
    VFC = de_date(VFC)
    non_VIC = de_date(non_VIC)
    non_sec_VFC = de_date(non_sec_VFC)
    non_sec_non_VIC = de_date(non_sec_non_VIC)
    
    log.info("Fetch security commit")
    label1s = [VIC, VIC, VIC, VIC, VIC]
    label0s = [VFC, non_VIC, [i + j for i, j in zip(VFC, non_VIC)], non_sec_non_VIC, [i + j for i, j in zip(non_sec_VFC, non_sec_non_VIC)]]
    
    log.info("To Dataset")
    try:
        to_dataset(project, output_folder, label0s, label1s, params.workers)     
        log.info("Complete!")
    except Exception as e:
        log.error(traceback.format_exc())
        shutil.rmtree('temp')   

if __name__ == "__main__":
    log.info("Start!!")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="label")
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--continue_run", action="store_true")
    parser.add_argument("--unsplit_only", action="store_true")
    params = parser.parse_args()
    run(params)
