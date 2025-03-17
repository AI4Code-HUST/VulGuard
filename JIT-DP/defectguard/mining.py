import os
from .utils.logger import logger
from .crawler import Pipeline
from argparse import Namespace

def mining(params):
    logger("Start DefectGuard")

    # create save folders
    dg_cache_path = f"{params.dg_save_folder}/dg_cache"
    folders = ["save", "repo", "dataset"]
    if not os.path.exists(dg_cache_path):
        os.mkdir(dg_cache_path)
    for folder in folders:
        if not os.path.exists(os.path.join(dg_cache_path, folder)):
            os.mkdir(os.path.join(dg_cache_path, folder))

    # User's input handling
    cfg = {
        "mode": params.mode,
        "repo_owner": params.repo_owner,
        "repo_name": params.repo_name,
        "repo_path": params.repo_path,
        "repo_clone_url": params.repo_clone_url,
        "repo_clone_path": params.repo_clone_path,
        "repo_language": [params.repo_language],
        "repo_save_path": f"{dg_cache_path}/save",
        "extractor_save": True,
        "extractor_reextract": params.reextract,
        "create_dataset": True,
        "pyszz_path": params.pyszz_path,
        "dataset_save_path": f"{dg_cache_path}/dataset",
        "processor_save": True,
        "num_commits_per_file": 5000,
        "szz": params.szz,
        "workers": params.workers
    }

    if params.mode == "remote":
        if cfg["repo_clone_path"] is None:
            cfg["repo_clone_path"] = f"{dg_cache_path}/repo"
        if cfg["repo_clone_url"] is None:
            cfg["repo_clone_url"] = f"https://github.com/{params.repo_owner}/{params.repo_name}.git"
        cfg["extractor_check_uncommit"] = False
    else:
        cfg["extractor_check_uncommit"] = params.uncommit

    cfg = Namespace(**cfg)    
    crawler = BasicPipeline(cfg)
    crawler.set_repo(cfg)
    crawler.run()
