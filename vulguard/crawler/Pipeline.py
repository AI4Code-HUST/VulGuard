import argparse, os
from .Miner import Miner
from .Extractor import Extractor
from .szz.main import run as SZZ
from .Labeler import Labeler
import pandas as pd

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def run(params):
    print("Preparation!")
    os.system("export GIT_OPTIONAL_LOCKS=0")
    lock_file = f"{params.repo_path}/{params.repo_name}/.git/config.lock"

    if os.path.exists(lock_file):
        os.remove(lock_file)
        print(f"Deleted lock file: {lock_file}")
    else:
        print(f"No lock file found at: {lock_file}")
        
    print("Miner!")
    print("=" * 20)
    miner = Miner(params)
    filtered_raw_commits = miner.run()
    
    
    print("=" * 20)
    df = pd.read_json(f"{params.dataset_save_path}/commit_ids_{params.repo_name}.jsonl", orient="records", lines=True)
    filtered_raw_commits = list(df["commit_id"])
    
    print("Extractor!")
    print("=" * 20)
    extractor = Extractor(params)
    extractor.run(filtered_raw_commits)
    
    print("=" * 20)
    
    print("SZZ!")
    print("=" * 20)
    szz_cfg = {
        "input_jsonl": params.vfc_file if params.vfc_file is not None else f"{params.dataset_save_path}/vfc_{params.repo_name}.jsonl",
        "save_path": params.dataset_save_path,
        "conf": params.szz,
        "num_core": params.workers,
        "repo_name": params.repo_name,
        "repos_dir": params.repo_path,
    }
    szz_cfg = argparse.Namespace(**szz_cfg)
    SZZ(szz_cfg)
    print("=" * 20)

    print("Labeler!")
    print("=" * 20)
    label_cfg = {
        "repo_name": params.repo_name,
        "save_path": params.dataset_save_path,
        "szz": params.szz,
    }
    label_cfg = argparse.Namespace(**label_cfg)
    labeler = Labeler(label_cfg)
    labeler.run()
    print("=" * 20)

# if __name__ == "__main__":
#     main()
