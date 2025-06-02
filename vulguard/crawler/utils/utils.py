from typing import List, Dict, Generator
# import logging as log
import json, os, uuid
import re
from pathlib import Path


STRONG_VUL = re.compile(r'(?i)(denial.of.service|remote.code.execution|\bopen.redirect|OSVDB|\bXSS\b|\bReDoS\b|\bNVD\b|malicious|x−frame−options|attack|cross.site|exploit|directory.traversal|\bRCE\b|\bdos\b|\bXSRF\b|clickjack|session.fixation|hijack|advisory|insecure|security|\bcross−origin\b|unauthori[z|s]ed|infinite.loop)')
MEDIUM_VUL =re.compile(r'(?i)(authenticat(e|ion)|bruteforce|bypass|constant.time|crack|credential|\bDoS\b|expos(e|ing)|hack|harden|injection|lockout|overflow|password|\bPoC\b|proof.of.concept|poison|privelage|\b(in)?secur(e|ity)|(de)?serializ|spoof|timing|traversal)')
def is_vfc(file_message: str) -> int:
    m = STRONG_VUL.search(file_message)
    n = MEDIUM_VUL.search(file_message)
    return 1 if m or n else 0 

def split_sentence(sentence):
    sentence = sentence.replace('.', ' . ').replace('_', ' ').replace('@', ' @ ')\
        .replace('-', ' - ').replace('~', ' ~ ').replace('%', ' % ').replace('^', ' ^ ')\
        .replace('&', ' & ').replace('*', ' * ').replace('(', ' ( ').replace(')', ' ) ')\
        .replace('+', ' + ').replace('=', ' = ').replace('{', ' { ').replace('}', ' } ')\
        .replace('|', ' | ').replace('\\', ' \ ').replace('[', ' [ ').replace(']', ' ] ')\
        .replace(':', ' : ').replace(';', ' ; ').replace(',', ' , ').replace('<', ' < ')\
        .replace('>', ' > ').replace('?', ' ? ').replace('/', ' / ')
    sentence = ' '.join(sentence.split())
    return sentence

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

def load_jsonl(file_path: str):
    with open(file_path, "r") as f:
        for line in f:
            yield json.loads(line)

def read_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_chunk_jsonl(file_path: str, start: int, end: int):
    with open(file_path, 'r') as file:
        file.seek(start)
        if start != 0:
            file.readline() 
        while file.tell() < end:
            line = file.readline().strip()
            if not line:
                break
            yield json.loads(line)


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