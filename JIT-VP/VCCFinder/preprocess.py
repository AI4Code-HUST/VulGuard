print("*** IMPORT ***")
import os, sys, shutil
import numpy as np
import sklearn.preprocessing
import re
from pathlib import Path
import json
from tqdm import tqdm
print("***************")

PRE_SALLY_DIR = os.path.join(Path(__file__).parent, "pre_sally")
POST_SALLY_DIR = os.path.join(Path(__file__).parent, "post_sally")
WORKSPACE = os.path.join(Path(__file__).parent, "workspace")

if os.path.exists(PRE_SALLY_DIR):
    shutil.rmtree(PRE_SALLY_DIR)
if os.path.exists(POST_SALLY_DIR):
    shutil.rmtree(POST_SALLY_DIR)
if os.path.exists(WORKSPACE):
    shutil.rmtree(WORKSPACE)
    

vcc_file = f"PATH/HERE/SETUP1-FFmpegvcc-features-train.jsonl"
patch_file = f"PATH/HERE/SETUP1-FFmpeg-simcom-train.jsonl" 
        

def find_files(regex_pattern: str, folder: str):
    pattern = re.compile(regex_pattern)
    matching_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if pattern.match(file):
                matching_files.append(os.path.join(root, file))
    return matching_files

def get_first_digit(x):
    """
    Value -> bin value
    10 possible bins
    x should be < 100
    """
    x = int(x)
    if x < 0:
        return 0
    x = str(x)
    if len(x) == 1:  # less than 10 ?
        return 0
    else:
        return int(x[0])
    
def yield_jsonl(file):
    with open(file, "r") as f:
        for line in f:
            yield json.loads(line)
    
def load_jsonl(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    return data

KEYS = [
    "commit_id", "addition", "deletion", "hunk_count", "kw_do", "kw_if", "kw_asm", "kw_for", "kw_int", "kw_new", "kw_try", "kw_auto", "kw_bool", 
    "kw_case", "kw_char", "kw_else", "kw_enum", "kw_free", "kw_goto", "kw_long", "kw_this", "kw_true", "kw_void", "kw_alloc", "kw_break", "kw_catch", 
    "kw_class", "kw_const", "kw_false", "kw_float", "kw_short", "kw_throw", "kw_union", "kw_using", "kw_while", "kw_alloca", "kw_calloc", "kw_delete", 
    "kw_double", "kw_extern", "kw_friend", "kw_inline", "kw_malloc", "kw_public", "kw_return", "kw_signed", "kw_sizeof", "kw_static", "kw_struct", 
    "kw_switch", "kw_typeid", "kw_default", "kw_mutable", "kw_private", "kw_realloc", "kw_typedef", "kw_virtual", "kw_wchar_t", "kw_continue", 
    "kw_explicit", "kw_operator", "kw_register", "kw_template", "kw_typename", "kw_unsigned", "kw_volatile", "kw_namespace", "kw_protected", "kw_const_cast", 
    "kw_static_cast", "kw_dynamic_cast", "kw_reinterpret_cast", "author_contributions_percent", "past_changes", "future_changes", "past_different_authors", 
    "future_different_authors"
]

def main(vcc_file, patch_file):
    # vcc_files = find_files(r"simcom-extracted-all-FFmpeg-start-\d+-end-\d+[.]jsonl$","/mnt/e/NewCrawler/output/extracted/FFmpeg")
    # patch_files = find_files(r"^simcom-extracted-all-FFmpeg-start-\d+-end-\d+[.]jsonl$","/mnt/e/NewCrawler/output/extracted/FFmpeg")
    
    print("START")
    out_dir = WORKSPACE
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(PRE_SALLY_DIR):
        os.makedirs(PRE_SALLY_DIR)
    if not os.path.exists(POST_SALLY_DIR):
        os.makedirs(POST_SALLY_DIR)
    if not os.path.exists(WORKSPACE):
        os.makedirs(WORKSPACE)

    print("*** MAP DATA ***")
    mapping = {key: [] for key in KEYS}
    labels = {}
    # for vcc_file in vcc_files:
    data = load_jsonl(vcc_file)    
    for datum in data:
        labels[datum["commit_id"]] = datum["label"]
        for key in datum.keys():
            if key in ["author", "files", "label"]:
                continue
            mapping[key].append(datum[key])
    del data
    print("*****************")
    
    print("*** BUILD SCALERS ***")
    normalisers = dict()
    for feat in mapping.keys():
        if feat == "commit_id":
            continue
        scaler = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=10, output_distribution='uniform'
        )
        scaler.fit(
            np.array(mapping[feat]).reshape(-1, 1)
        )
        normalisers[feat] = lambda x, scaler=scaler: int(
            scaler.transform(np.array(x).reshape(-1, 1)) * 100
        )
    print("*******************")
    
    def commit_to_txt(commit_id: int, patch: str, message: str):
        index = mapping["commit_id"].index(commit_id)
        res = list()

        for feat in mapping.keys():
            if feat == "commit_id":
                continue
            val = mapping[feat][index]
            if val:
                normalised = normalisers[feat](val)
                my_bin = get_first_digit(normalised)
                res.append(f"{feat}::{my_bin}")
        
        # Extract text from patch
        res.extend(patch.split('\n'))
        res.extend(message.split('\n'))
        return ' '.join(res)

    def commit_to_file(commit_id: int, patch: str, message: str, dir_location=PRE_SALLY_DIR):
        string = commit_to_txt(commit_id, patch, message, mapping).encode('utf-8', errors='ignore').decode("utf-8")
        with open(os.path.join(dir_location, str(commit_id)), 'w') as f:
            f.write(string + "\n")

    # for patch_file in patch_files:
    for commit in yield_jsonl(patch_file):
        commit_id = commit["commit_id"]
        patch = commit["code_change"]
        message = commit["messages"]
        
        commit_to_file(commit_id, patch, message)
        
    print("*** LINES TO VECTOR ***")
    regex = r"[0-9a-zA-Z]{40}"
    files = find_files(regex, PRE_SALLY_DIR)
    cmd = "sally -i lines -o libsvm --vect_embed bin -d' ' -g tokens " + WORKSPACE + "/* " + WORKSPACE + "/one_line.libsvm"
    name = vcc_file.rsplit("/")[-1].split(".")[0]
    for file in tqdm(files):
        file = file.rsplit("/")[-1]
        shutil.copy(os.path.join(PRE_SALLY_DIR, file), WORKSPACE)
        os.system(cmd)
        with open(WORKSPACE + '/one_line.libsvm', 'r') as f:
            post_sally_lines = f.readlines()
        assert len(post_sally_lines) == 1
        
        label = labels[file]
        new_line = (re.sub('#[^\n]+', "#" + file, post_sally_lines[0])).replace('\n', '')
        new_line = re.sub('^\S+\s', str(label) + " ", new_line)
        
        
        with open(os.path.join(POST_SALLY_DIR, f"{name}.libsvm"), "a") as f:
            f.write(new_line + "\n")
        
        os.remove(os.path.join(PRE_SALLY_DIR, file))
        os.remove(os.path.join(WORKSPACE, file))
        os.remove(os.path.join(WORKSPACE, "one_line.libsvm"))
    print("***********************")
    
if __name__ == "__main__":
    main(vcc_file, patch_file)