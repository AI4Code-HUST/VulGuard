import argparse
import json
import logging as log
import os
from typing import List
import yaml
from typing import Dict
from .szz.ag_szz import AGSZZ
from .szz.aszz.a_szz import ASZZ
from .szz.b_szz import BaseSZZ
from .szz.util.check_requirements import check_requirements
from .szz.dfszz.df_szz import DFSZZ
from .szz.l_szz import LSZZ
from .szz.ma_szz import MASZZ, DetectLineMoved
from .szz.r_szz import RSZZ
from .szz.ra_szz import RASZZ
from .szz.pd_szz import PyDrillerSZZ
from .szz.vszz.v_szz import VSZZ
from .szz.common.issue_date import parse_issue_date
from pathlib import Path
from multiprocessing import Manager
import concurrent.futures as cf
from .options import Options
from traceback import format_exc

def create_log_handler(worker_id: int):
    # Construct the log filename based on the core ID
    log_filename = f'log/logs_core_{worker_id}.log'
    
    # Create a logger with the core-specific name
    logger = log.getLogger(name=f'worker_{worker_id}')
    logger.setLevel(log.INFO)  # Set the logging level to INFO
    
    # Create a file handler for logging
    fileHandler = log.FileHandler(log_filename)
    fileHandler.setLevel(log.INFO)  # Set the handler level to INFO

    # Define the log message format
    formatter = log.Formatter('%(asctime)s :: %(funcName)s - %(levelname)s :: %(message)s')
    fileHandler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fileHandler)

    # Set specific logging levels for other modules, if needed
    log.getLogger('pydriller').setLevel(log.WARNING)

    # Log the initialization message
    logger.info(f'Logging initialized for core {worker_id}')

    return logger

def main(input_json: str, out_json: str, conf: Dict, repos_dir: str, worker_id: int = 0):
    logger = create_log_handler(worker_id)  # Initialize logging for each core
    
    bugfix_commits = []
    with open(input_json, 'r') as in_file:
        for line in in_file:
            bugfix_commits.append(json.loads(line))
    tot = len(bugfix_commits)

    with open(out_json, 'a') as out:
        for i, commit in enumerate(bugfix_commits):
            bug_inducing_commits = set()
            repo_name = commit['Repository']
            repo_url = f'https://test:test@github.com/{repo_name}.git'  # using test:test as git login to skip private repos during clone
            fix_commit = commit['commit_id']

            logger.info(f'{i + 1} of {tot}: {repo_name} {fix_commit}')
            
            issue_date = None
            if conf.get('issue_date_filter', None):
                issue_date = parse_issue_date(commit)
            
            szz_name = conf['szz_name']
            if szz_name == 'b':
                b_szz = BaseSZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir, logger=logger)
                imp_files = b_szz.get_impacted_files(fix_commit_hash=fix_commit, file_ext_to_parse=conf.get('file_ext_to_parse'), only_deleted_lines=True)
                bug_inducing_commits = b_szz.find_bic(fix_commit_hash=fix_commit,
                                            impacted_files=imp_files,
                                            issue_date_filter=conf.get('issue_date_filter'),
                                            issue_date=issue_date)
            elif szz_name == 'ag':
                ag_szz = AGSZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = ag_szz.get_impacted_files(fix_commit_hash=fix_commit, file_ext_to_parse=conf.get('file_ext_to_parse'), only_deleted_lines=True)
                bug_inducing_commits = ag_szz.find_bic(fix_commit_hash=fix_commit,
                                            impacted_files=imp_files,
                                            max_change_size=conf.get('max_change_size'),
                                            issue_date_filter=conf.get('issue_date_filter'),
                                            issue_date=issue_date)
            elif szz_name == 'ma':
                ma_szz = MASZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = ma_szz.get_impacted_files(fix_commit_hash=fix_commit, file_ext_to_parse=conf.get('file_ext_to_parse'), only_deleted_lines=True)
                bug_inducing_commits = ma_szz.find_bic(fix_commit_hash=fix_commit,
                                            impacted_files=imp_files,
                                            max_change_size=conf.get('max_change_size'),
                                            detect_move_from_other_files=DetectLineMoved(conf.get('detect_move_from_other_files')),
                                            issue_date_filter=conf.get('issue_date_filter'),
                                            issue_date=issue_date,
                                            filter_revert_commits=conf.get('filter_revert_commits', False))
            elif szz_name == 'r':
                r_szz = RSZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = r_szz.get_impacted_files(fix_commit_hash=fix_commit, file_ext_to_parse=conf.get('file_ext_to_parse'), only_deleted_lines=True)
                bug_inducing_commits = r_szz.find_bic(fix_commit_hash=fix_commit,
                                            impacted_files=imp_files,
                                            max_change_size=conf.get('max_change_size'),
                                            detect_move_from_other_files=DetectLineMoved(conf.get('detect_move_from_other_files')),
                                            issue_date_filter=conf.get('issue_date_filter'),
                                            issue_date=issue_date,
                                            filter_revert_commits=conf.get('filter_revert_commits', False))
            elif szz_name == 'l':
                l_szz = LSZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = l_szz.get_impacted_files(fix_commit_hash=fix_commit, file_ext_to_parse=conf.get('file_ext_to_parse'), only_deleted_lines=True)
                bug_inducing_commits = l_szz.find_bic(fix_commit_hash=fix_commit,
                                            impacted_files=imp_files,
                                            max_change_size=conf.get('max_change_size'),
                                            detect_move_from_other_files=DetectLineMoved(conf.get('detect_move_from_other_files')),
                                            issue_date_filter=conf.get('issue_date_filter'),
                                            issue_date=issue_date,
                                            filter_revert_commits=conf.get('filter_revert_commits', False))
            elif szz_name == 'ra':
                ra_szz = RASZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = ra_szz.get_impacted_files(fix_commit_hash=fix_commit, file_ext_to_parse=conf.get('file_ext_to_parse'), only_deleted_lines=True)
                bug_inducing_commits = ra_szz.find_bic(fix_commit_hash=fix_commit,
                                            impacted_files=imp_files,
                                            max_change_size=conf.get('max_change_size'),
                                            detect_move_from_other_files=DetectLineMoved(conf.get('detect_move_from_other_files')),
                                            issue_date_filter=conf.get('issue_date_filter'),
                                            issue_date=issue_date,
                                            filter_revert_commits=conf.get('filter_revert_commits', False))
            elif szz_name == 'pd':
                pd_szz = PyDrillerSZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = pd_szz.get_impacted_files(fix_commit_hash=fix_commit, file_ext_to_parse=conf.get('file_ext_to_parse'), only_deleted_lines=True)
                bug_inducing_commits = pd_szz.find_bic(fix_commit_hash=fix_commit,
                                                    impacted_files=imp_files,
                                                    issue_date_filter=conf.get('issue_date_filter'),
                                                    issue_date=issue_date)
            elif szz_name == 'a':
                a_szz = ASZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                bug_inducing_commits = a_szz.start(fix_commit_hash=fix_commit, commit_issue_date=issue_date, **conf)

            elif szz_name == 'df':
                df_szz = DFSZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                bug_inducing_commits = df_szz.start(fix_commit_hash=fix_commit, commit_issue_date=issue_date, **conf)
            elif szz_name == "v":
                v_szz = VSZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir, ast_map_path=conf.get('ast_map_path'), logger=logger)
                imp_files = v_szz.get_impacted_files(fix_commit_hash=fix_commit, file_ext_to_parse=conf.get('file_ext_to_parse'), only_deleted_lines=True)
                bug_inducing_commits = v_szz.find_bic(fix_commit_hash=fix_commit,
                                                impacted_files=imp_files,
                                                ignore_revs_file_path=None,
                                                issue_date_filter=conf.get('issue_date_filter'),
                                                issue_date=issue_date)

            else:
                logger.info(f'SZZ implementation not found: {szz_name}')
                exit(-3)

            logger.info(f"result: {bug_inducing_commits}")
            if szz_name == "v":
                bugfix_commits[i]["detected_id"] = [bic for bic in bug_inducing_commits if bic]
            else:
                bugfix_commits[i]["detected_id"] = [bic.hexsha for bic in bug_inducing_commits if bic]

            out.write(json.dumps(commit) + '\n')

    logger.info(f"results saved in {out_json}")
    logger.info("+++ DONE +++")
    
def split_list(lst, n):
    if len(lst) < n:
        return [lst]
    chunk_size = len(lst) // n
    remainder = len(lst) % n
    result = [lst[i*chunk_size:(i+1)*chunk_size] for i in range(n)]
    result[-1].extend(lst[n*chunk_size:])
    return result

def split_json(input_json: str, input_name: str, temp_dir: str, num_core: int = 1):
    bugfix_commits = []
    with open(input_json, 'r') as in_file:
        for line in in_file:
            bugfix_commits.append(json.loads(line))
    sublists = split_list(bugfix_commits, num_core)        
    out_file = []
    for id in range(num_core):
        out_file.append(f'{temp_dir}/{input_name}_{id}.jsonl')
        with open(out_file[id], 'w') as out:
            for line in sublists[id]:
                out.write(json.dumps(line) + '\n')
    return out_file

def merge_json(output_jsons: List[int], output_name: str, out_dir: str, num_core: int = 1):
    output = []
    for i in range(num_core):
        with open(output_jsons[i], 'r') as f:
            for line in f:
                output.append(json.loads(line))
    with open(f'{out_dir}/{output_name}.jsonl', "w") as f:
        for line in output:
            f.write(json.dumps(line) + "\n")
    return output

def run(args):
    try: 
        with open(args.conf_file, 'r') as f:
            conf = yaml.safe_load(f)

        log.info(f"parsed conf yml '{args.conf_file}': {conf}")
        szz_name = conf['szz_name']
        input_name = args.input_json.rsplit('/', 1)[-1].split('.')[0]

        out_dir = Options.SZZ_OUTPUT + f"/{args.repo_name}"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        temp_dir = Options.TEMP_WORKING_DIR
        os.makedirs(temp_dir, exist_ok=True)
        __temp_dir = mkdtemp(dir=os.path.join(Options.PYSZZ_HOME, Options.TEMP_WORKING_DIR))

        log_dir = Options.SZZ_LOG_DIR
        os.makedirs(log_dir, exist_ok=True)

        conf_file_name = Path(args.conf_file).name.split('.')[0]
        input_jsons = split_json(args.input_json, input_name, temp_dir, args.num_core)
        output_jsons = [os.path.join(temp_dir, f'bic_{conf_file_name}_{input_name}_{id}.jsonl') for id in range(args.num_core)]

        if not szz_name:
            log.error('The configuration file does not define the SZZ name. Please, fix.')
            exit(-3)

        log.info(f'Launching {szz_name}-szz')

        manager = Manager()
        futures = []
        with cf.ProcessPoolExecutor(args.num_core) as pp:
            for id in range(args.num_core):
                futures.append(pp.submit(main, input_jsons[id], output_jsons[id], conf, args.repos_dir, id))
        for future in cf.as_completed(futures):
            future.result()

        output = merge_json(output_jsons, f'bic_{conf_file_name}_{input_name}.jsonl', out_dir, args.num_core)
    except:
        log.error(format_exc())    

if __name__ == "__main__":
    check_requirements()

    parser = argparse.ArgumentParser(description='USAGE: python main.py <bugfix_commits.json> <conf_file path> <repos_directory(optional)>\n* If <repos_directory> is not set, pyszz will download each repository')
    parser.add_argument('input_jsonl', type=str, help='/path/to/bug-fixes.jsonl')
    parser.add_argument('conf_file', type=str, help='/path/to/configuration-file.yml')
    parser.add_argument('repos_dir', type=str, nargs='?', help='/path/to/repo-directory')
    parser.add_argument('num_core', type=int, default=1, help='number of workers, default = 1')
    parser.add_argument('repo_name', type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.input_json):
        log.error('invalid input json')
        exit(-2)
    if not os.path.isfile(args.conf_file):
        log.error('invalid conf file')
        exit(-2)
        
    run(args)