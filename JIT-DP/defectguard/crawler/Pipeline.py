import argparse, os
from .Miner import Miner
from .Extractor import Extractor
from .szz.main import run as SZZ

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def main():
    parser = argparse.ArgumentParser(add_help= False)
    parser.add_argument("--workers", type= int, default= 1, help="Number of parallel workers")
    parser.add_argument("--language", type= str, help="Language")
    parser.add_argument("--url", type=str, help= "Git clone url")
    parser.add_argument("--path", type=str, help= "Local Repo path", default= f"{DIR_PATH}/input")
    parser.add_argument("--repo_name", type=str, help="Repo name")
    parser.add_argument("--start", type=int, default=None, help= "First commit index")
    parser.add_argument("--end", type=int, default=None, help="Last commit index")

    params = parser.parse_args()
    run(params)

def run(params):
    print("Miner!")
    print("=" * 20)
    miner_cfg = {
        "input_path": params.repo_path,
        "output_path": params.repo_save_path,
        "url": params.repo_clone_url,
        "language": params.repo_language[0],
        "workers": params.workers
    }
    miner_cfg = argparse.Namespace(**miner_cfg)
    miner = Miner(miner_cfg)
    out_file = miner.run()
    print("=" * 20)
    
    print("Extractor!")
    print("=" * 20)
    ext_cfg = {
        "repo_name": params.repo_name,
        "continue_run": False,
        "save_path": f"{params.dataset_save_path}/extracted/{params.repo_name}"
    }
    ext_cfg = argparse.Namespace(**ext_cfg)
    extractor = Extractor(ext_cfg)
    extractor.run()
    print("=" * 20)
    
    print("SZZ!")
    print("=" * 20)
    szz_cfg = {
        "input_jsonl": f"{params.dataset_save_path}/extracted/{params.repo_name}/bugfixes-{params.repo_name}.jsonl",
        "conf": "./defectguard/crawler/szz/conf/bszz.yml",
        "repos_dir": params.repo_path,
        "num_core": params.workers,
        "repo_name": params.repo_name
    }
    szz_cfg = argparse.Namespace(**szz_cfg)
    SZZ(szz_cfg)
    print("=" * 20)
    
    print("Labeler!")
    print("=" * 20)
    label_cfg = {
        "input_folder": f"./defectguard/crawler/szz/out/{params.repo_name}",
        "output_folder": f"{params.dataset_save_path}/data",
        "project": params.repo_name,
        "workers": params.workers
    }
    label_cfg = argparse.Namespace(**label_cfg)
    SZZ(label_cfg)
    print("=" * 20)

if __name__ == "__main__":
    main()
