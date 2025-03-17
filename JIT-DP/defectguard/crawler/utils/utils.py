from typing import List, Dict
import logging as log
import json, os, uuid
import re
from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent
DEFAULT_LOG = f"{PARENT_DIR}/log"
DEFAULT_INPUT = f'{PARENT_DIR}/input'
DEFAULT_EXTRACTED_OUTPUT = f'{PARENT_DIR}/output/extracted'
DEFAULT_DATA_OUTPUT = f"{PARENT_DIR}/output/dataset"

EXTRACTED_FILE_NAME_PATERN = "{}-.*[.]jsonl$"
SIMCOM_PATERN = EXTRACTED_FILE_NAME_PATERN.format("simcom")
DEEPJIT_PATERN = EXTRACTED_FILE_NAME_PATERN.format("deepjit")
SECURITY_PATERN = EXTRACTED_FILE_NAME_PATERN.format("security")
FEATURES_PATERN = EXTRACTED_FILE_NAME_PATERN.format("features")
VCCFINDER_PATERN = EXTRACTED_FILE_NAME_PATERN.format("vcc-features")

TRUSTED_LABEL_PATERN = "^T_{}.*[.]jsonl$"
SEMI_TRUSTED_LABEL_PATERN = "^ST_{}.*[.]jsonl$"

if not os.path.exists(DEFAULT_EXTRACTED_OUTPUT):
    os.makedirs(DEFAULT_EXTRACTED_OUTPUT)
if not os.path.exists(DEFAULT_DATA_OUTPUT):
    os.makedirs(DEFAULT_DATA_OUTPUT)

def create_console_log_handler(name: str) -> log.Logger:  
    logger = log.getLogger(name=name)
    logger.setLevel(log.INFO)
    logger.handlers.clear() 
    consoleHandler = log.StreamHandler() 
    consoleHandler.setLevel(log.INFO)
    formatter = log.Formatter('%(asctime)s :: %(funcName)s - %(levelname)s :: %(message)s')
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    log.info(f"Logging initialized")
    return logger

def create_log_handler(name: str) -> log.Logger:
    if not os.path.exists(DEFAULT_LOG):
        os.mkdir(DEFAULT_LOG)
        
    log_filename = f'{DEFAULT_LOG}/{name}'    
    logger = log.getLogger(name=name)
    logger.setLevel(log.INFO)  
    fileHandler = log.FileHandler(log_filename)
    fileHandler.setLevel(log.INFO) 
    formatter = log.Formatter('%(asctime)s :: %(funcName)s - %(levelname)s :: %(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    log.getLogger('pydriller').setLevel(log.WARNING)
    logger.info(f'Logging initialized for {name}')
    return logger

def save_json(data: Dict, output_file: str) -> None:
    with open(output_file, 'w') as f:
        json.dump(data, f)

def load_json(file_path: str) -> Dict:
    with open(file_path, "r") as f:
        out_dict = json.load(f)
    return out_dict

def save_jsonl(data: List[Dict], output_file: str) -> None:
    with open(output_file, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

def append_jsonl(data: List[Dict], output_file: str) -> None:
    with open(output_file, 'a') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

def load_jsonl(file_path: str) -> Dict:
    with open(file_path, "r") as f:
        for line in f:
            yield json.loads(line)

def read_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_chunk_jsonl(file_path: str, start: int, end: int) -> Dict:
    with open(file_path, 'r') as file:
        file.seek(start)
        if start != 0:
            file.readline() 
        while file.tell() < end:
            line = file.readline().strip()
            if not line:
                break
            yield json.loads(line)

def split_list(lst: List, part: int) -> List[List]:
    if len(lst) < part:
        return [lst]
    chunk_size = len(lst) // part
    remainder = len(lst) % part
    result = [lst[i*chunk_size:(i+1)*chunk_size] for i in range(n)]
    result[-1].extend(lst[part*chunk_size:])
    return result

def generate_id():
    return uuid.uuid1()

def find_files(regex_pattern: str, folder: str) -> List[str]:
    pattern = re.compile(regex_pattern)
    matching_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if pattern.match(file):
                matching_files.append(os.path.join(root, file))
    return matching_files