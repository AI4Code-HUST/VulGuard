from .utils.utils import create_dg_cache
from .utils.logger import logger
from .crawler import Pipeline
from argparse import Namespace

def mining(params):
    logger("Start VulGuard")
    print("Start VulGuard")
    dg_cache_path = create_dg_cache(params.dg_save_folder)
    
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
