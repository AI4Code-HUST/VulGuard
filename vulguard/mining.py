import os
from .utils.logger import logger
from .crawler import Pipeline
from argparse import Namespace

def mining(params):
    logger("Start VulGuard")
    print("Start VulGuard")

    # create save folders
    dg_cache_path = f"{params.dg_save_folder}/dg_cache"
    folders = ["save", "repo", "dataset"]
    if not os.path.exists(dg_cache_path):
        os.mkdir(dg_cache_path)
    for folder in folders:
        if not os.path.exists(os.path.join(dg_cache_path, folder)):
            os.mkdir(os.path.join(dg_cache_path, folder))

    # User's input handling
    assert params.repo_name is not None, "Please provide the miner with repo_name"
    
    cfg = {
        "mode": params.mode,
        "repo_name": params.repo_name,
        "repo_path": params.repo_path,
        "repo_clone_url": params.repo_clone_url,
        "repo_clone_path": params.repo_clone_path,
        "repo_language": [params.repo_language],
        "repo_save_path": f"{dg_cache_path}/save/{params.repo_name}",
        "dataset_save_path": f"{dg_cache_path}/dataset/{params.repo_name}",
        "szz": params.szz,
        "workers": params.workers,
        "start": params.start,
        "end": params.end,
        "logger": logger,
        "vfc_file": params.vfc_file
    }

    cfg = Namespace(**cfg)    
    Pipeline.run(cfg)
