from git import Repo, Commit
from typing import Dict, List
from datetime import datetime, timezone
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import traceback
import logging as logger
import os, shutil
import heapq

from .utils.aggregator import *
from .utils.line_parser import *
from .utils.utils import *

EXT2LANG = {
    "py": "Python",
    "java": "Java",
    "cpp": "C++",
    "c": "C",
    "js": "JavaScript",
    "rb": "Ruby",
    "swift": "Swift",
    "go": "Go",
    "rs": "Rust",
    "ts": "TypeScript",
    "php": "PHP",
    "cs": "C#",
    "h": "C",
    # Add more extensions and programming languages as needed
}

class Miner:
    def __init__(self, params):
        self.url = params.url
        self.repo_path = params.path
        self.workers = params.workers
        self.start = params.start
        self.end = params.end
        
        self.num_commits_per_files = 1000
        self.logger = create_log_handler("logs_miner_main.log")
        self.repo_name = None

        if self.url is not None:
            if not os.path.exists(self.repo_path):
                os.mkdir(self.repo_path)
            
            self.repo_name = os.path.basename(self.url)
            if self.repo_name.endswith('.git'):
                self.repo_name = self.repo_name[:-4]

            self.repo_path = f"{self.repo_path}/{self.repo_name}"

            try:
                Repo.clone_from(self.url, self.repo_path)
            except FileExistsError:
                self.logger.info("File existed")
            except Exception as e:
                self.logger.info(f"{e}")         
        try:
            self.repo = Repo(self.repo_path)
            self.languages = params.language
            # self.logger.info(params.language)
        except Exception as e:
            self.logger.error(f"Catch error: {e}")
            self.logger.error(f"Cannot find {self.repo_path}")

        if self.repo_name is None:
            self.repo_name = self.repo.remotes.origin.url.rstrip('/').split('/')[-1].replace('.git', '')
        
        self.save_path = f"{DEFAULT_OUTPUT}/{self.repo_name}"        
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
    
    def run(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        result = self.process_parallel()
        
        if self.start is not None and self.end is not None:
            out_file = f"{self.save_path}/extracted-all-{self.repo_name}-start-{self.start}-end-{self.end}.jsonl"
        elif self.start is not None:
            out_file = f"{self.save_path}/extracted-all-{self.repo_name}-start-{self.start}.jsonl"
        elif self.end is not None:
            out_file = f"{self.save_path}/extracted-all-{self.repo_name}-end-{self.end}.jsonl"
        else:
            out_file = f"{self.save_path}/extracted-all-{self.repo_name}.jsonl"
        save_jsonl(result, out_file)
        return out_file

    def process_one_commit(self, commit_id: str, logger: logger.Logger) -> Dict:
        "git show {commit_id} --name-only --pretty=format:%H%n%P%n%an%n%ct%n%s%n%B%n[MODIFIED]"
        """
        Example output:
        Commit ID:      76137d3f1906af4afc18ccd62336d85cbc0c56a4
        Parents ID:     70ce2ed39fdb4057392ca9a584e1e47938e27ef3
        Authour:        Mr-Duo
        Date:           1724249410
        Subject:        stuff
        Message:        stuff
        [MODIFIED]
        Modified Files: defectguard/JITCrawler/core/utils/utils.py
        """

        show_msg = self.repo.git.show(commit_id, name_only=True, pretty='format:%H%n%P%n%an%n%ct%n%s%n%B%n[MODIFIED]').splitlines()
        files_index = show_msg.index('[MODIFIED]')
        subject = show_msg[4]
        head = show_msg[:5]
        commit_msg = show_msg[5:files_index]

        parent_id = head[1]
        author = head[2]
        commit_date = head[3]
        commit_msg = " ".join(commit_msg)

        "git show {commit_id} --pretty=format: --unified=999999999"
        """
        Example output:
        diff --git a/libavcodec/riscv/h264dsp_rvv.S b/libavcodec/riscv/h264dsp_rvv.S
        index a38bf7ef1d..0e08de43e4 100644
        --- a/libavcodec/riscv/h264dsp_rvv.S
        +++ b/libavcodec/riscv/h264dsp_rvv.S
        @@ -1,332 +1,327 @@
        [CODE CHANGES]
        """

        raw_diff_log = self.repo.git.show(commit_id, pretty='format:', unified=999999999).splitlines()
        unfiltered_diff_log = split_diff_log(raw_diff_log)
        diff_log = [log for log in unfiltered_diff_log if log[0][:10] == "diff --git"]
        # logger.info(raw_diff_log)
        commit_diff = {}
        commit_blame = {}
        files = []
        for log in diff_log:
            try:
                files_diff = aggregator(parse_lines(log))
            except:
                logger.error(f"Exception {e} : {log}")
            for file_diff in files_diff:                
                file_name_a = (
                    file_diff["from"]["file"]
                    if file_diff["rename"] or file_diff["from"]["mode"] != "0000000"
                    else file_diff["to"]["file"]
                )
                file_name_b = (
                    file_diff["to"]["file"]
                    if file_diff["rename"] or file_diff["to"]["mode"] != "0000000"
                    else file_diff["from"]["file"]
                )
                if file_diff["is_binary"] or len(file_diff["content"]) == 0:
                    continue

                if file_diff["from"]["mode"] == "0000000":
                    continue
                
                try:
                    file_extension = file_name_b.rsplit(".")[1].lower()
                except:
                    file_extension = None

                file_language = EXT2LANG.get(file_extension, None)
                if file_language is None or file_language.lower() != self.languages.lower():
                    continue

                "git blame -t -n -l {parent_id} '{file_name_a}'"
                """
                Example output:
                746f1ff36ac0d232687820fbde4e4efc79093af7   1 (Rémi Denis-Courmont 1664203942 +0300   1) /*
                746f1ff36ac0d232687820fbde4e4efc79093af7   2 (Rémi Denis-Courmont 1664203942 +0300   2)  * Copyright © 2022 Rémi Denis-Courmont.
                746f1ff36ac0d232687820fbde4e4efc79093af7   3 (Rémi Denis-Courmont 1664203942 +0300   3)  * Loosely based on earlier work copyrighted by Måns Rullgård, 2008.
                746f1ff36ac0d232687820fbde4e4efc79093af7   4 (Rémi Denis-Courmont 1664203942 +0300   4)  *
                746f1ff36ac0d232687820fbde4e4efc79093af7   5 (Rémi Denis-Courmont 1664203942 +0300   5)  * This file is part of FFmpeg.
                """

                file_blame_log = self.repo.git.blame(parent_id, file_name_a, t=True, n=True, l=True).splitlines()

                if not file_blame_log:
                    continue

                file_blame = get_file_blame(file_blame_log)
                commit_blame[file_name_b] = file_blame
                commit_diff[file_name_b] = file_diff
                files.append(file_name_b)
            
        if len(files) == 0:
            return None

        commit = {
            "commit_id": commit_id,
            "parent_id": parent_id,
            "subject": subject,
            "message": commit_msg,
            "author": author,
            "date": int(commit_date),
            "files": files,
            "diff": commit_diff,
            "blame": commit_blame,
        }
        return commit
    
    def process_multiple_commits(self, commit_ids: List[str], worker_id: int = 0) -> List[Dict]:
        extracted_commits_list = []
        log_file = f"logs_miner_{self.repo_name}_{worker_id}.log"
        logger = create_log_handler(log_file)
        # logger.info(commit_ids)
        for commit_id in tqdm(commit_ids, f"Thread {worker_id}"):            
            if len(extracted_commits_list) % self.num_commits_per_files == 0:
                file_id = generate_id()
                out_file = f"{self.save_path}/extracted-{self.repo_name}-{file_id}.jsonl"
            
            try: 
                extracted_commit = self.process_one_commit(commit_id, logger)
                if extracted_commit is not None:
                    extracted_commits_list.append(extracted_commit)
                    append_jsonl([extracted_commit], out_file)
            except Exception as e:
                logger.error(f"Exception {e} - Failed to mine {commit_id}")
                logger.error(traceback.format_exc())
        return worker_id, extracted_commits_list

    def process_parallel(self):
        self.commits = [commit.hexsha for commit in self.repo.iter_commits()]
        if self.start is not None and self.end is not None:
            self.commits = self.commits[self.start:self.end]
        elif self.start is not None:
            self.commits = self.commits[self.start:]
        elif self.end is not None:
            self.commits = self.commits[:self.end]
        self.commits.reverse()        
        # self.logger.info(self.commits)
        num_commits = len(self.commits)
        sublists = split_list(self.commits, self.workers)
        num_thread = self.workers
        
        futures = []
        results = []
        self.logger.info("Start processing")
        with ProcessPoolExecutor(max_workers=num_thread) as executor:
            for thread_id in range(num_thread):
                self.logger.info(f"Initiate thread {thread_id}")
                futures.append(executor.submit( self.process_multiple_commits, sublists[thread_id], thread_id))

            for future in as_completed(futures):
                result = future.result()
                self.logger.info(f"Thread {result[0]} completed!")
                results.append(result[1])

        final_results = list(heapq.merge(*results, key=lambda x: x["date"]))
        return final_results

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help= False)
    parser.add_argument("--workers", type= int, default= 1, help="Number of parallel workers")
    parser.add_argument("--language", type= str, help="Language")
    parser.add_argument("--url", type=str, help= "Git clone url")
    parser.add_argument("--input_path", type=str, help= "Parent directory of input repository", default= DEFAULT_INPUT)
    parser.add_argument("--output_path", type=str, help= "Output directory", default= DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=None, help= "First commit index")
    parser.add_argument("--end", type=int, default=None, help="Last commit index")

    params = parser.parse_args()
    miner = Miner(params)
    miner.run()
