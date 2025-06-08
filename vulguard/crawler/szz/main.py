import argparse
import json
import logging
import os
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
import concurrent.futures as cf
from traceback import format_exc

def main(commit, conf: Dict, repos_dir: str, logger):
    bug_inducing_commits = set()
    repo_name = commit['Repository']
    repo_url = f'https://test:test@github.com/{repo_name}.git'  # using test:test as git login to skip private repos during clone
    fix_commit = commit["commit_id"]
    logger.info(f'Repository {repo_name} - VFC: {fix_commit}')
    
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
    commit["VIC"] = [bic.hexsha for bic in bug_inducing_commits if bic]

    logger.info("+++ DONE +++")
    # print(commit)
    return commit

def get_logger():
    logging.basicConfig(
        level=logging.INFO,  # Set log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
        handlers=[
            logging.StreamHandler()               # Also log to console
        ]
    )

    # Example usage
    logger = logging.getLogger(__name__)
    return logger
    
def run(args):
    log = get_logger()
    log.disabled = True
    try: 
        conf_file = f"vulguard/crawler/szz/conf/{args.conf}.yml"
        with open(conf_file, 'r') as f:
            conf = yaml.safe_load(f)
            
        log.info(f"Parsed conf yml '{args.conf}': {conf}")
        szz_name = conf['szz_name']
        if not szz_name:
            log.error('The configuration file does not define the SZZ name. Please, fix.')
            exit(-3)

        log.info(f'Launching {szz_name}-szz')

        out_dir = args.save_path
        os.makedirs(out_dir, exist_ok=True)
        
        fix_commits = []
        with open(args.input_jsonl, "r") as f:
            for line in f:
                vfc = json.loads(line.strip())
                fix_commits.append(vfc)
               
        futures = []
        results = []

        with cf.ProcessPoolExecutor(max_workers=args.num_core) as pp:
            for vfc in fix_commits:
                futures.append(pp.submit(main, vfc, conf, args.repos_dir, log))

        for future in cf.as_completed(futures):
            results.append(future.result())

        # print(results)
        save_file = f"{args.save_path}/vic_{args.conf}_{args.repo_name}.jsonl"
        with open(save_file, "w") as f:
            for line in results:
                f.write(json.dumps(line) + "\n")
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
      
    run(args)