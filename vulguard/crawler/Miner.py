from git import Repo
from typing import Dict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging as logger
import os
import pandas as pd

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
        self.mode = params.mode
        self.repo_name = params.repo_name
        self.languages = [language.lower() for language in params.repo_language]
        
        self.workers = params.workers
        self.start = params.start if params.start is not None else 0
        self.end = params.end + 1 if params.start is not None else None
        self.logger = params.logger
        
        
        if self.mode == "local":
            self.repo_path = params.repo_path      
        elif self.mode == "remote":
            self.repo_clone_url = params.repo_clone_url 
            self.repo_path = params.repo_clone_path if params.repo_clone_path is not None else f"{params.repo_save_path}/{params.repo_name}"
            
            if not os.path.exists(self.repo_path):
                os.mkdir(self.repo_path)

            try:
                Repo.clone_from(self.url, self.repo_path)
            except FileExistsError:
                self.logger("File existed")
            except Exception as e:
                self.logger(f"{e}")     
        
        assert self.repo_path is not None, "Repo path must be provided"
        self.repo = Repo(f"{self.repo_path}/{self.repo_name}")

        self.save_path = params.dataset_save_path
        self.raw_save_path = f"{params.dataset_save_path}/raw"
        os.makedirs(self.raw_save_path, exist_ok=True)

    def run(self):
        self.repo.git.custom_environment(GIT_OPTIONAL_LOCKS='0')
        self.commits = [commit.hexsha for commit in self.repo.iter_commits()]
        self.commits = self.commits[self.start : self.end]
        self.commits.reverse()                
        
        futures = []
        results = []
        self.logger("Start processing")
        
        print("Start Mining:")
        with ProcessPoolExecutor(max_workers = self.workers) as executor:
            futures = {executor.submit(self.process_one_commit, commit, self.logger): commit for commit in self.commits}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing commits"):
                commit = futures[future]
                try:
                    results.append(future.result())
                    self.logger(f"Extracted raw commit {commit}!")
                except Exception as e:
                    self.logger(f"Failed to process commit {commit}: {e}")
        
        commit_df = pd.DataFrame(results)
        # print(commit_df)
        commit_df = commit_df[commit_df["keep"]==1]
        commit_df = commit_df.drop("keep", axis=1)
        commit_df = commit_df.sort_values(by="date")
        print("Mining Complete!")
        save_file = f"{self.save_path}/commit_ids_{self.repo_name}.jsonl"
        commit_df.to_json(save_file, orient="records", lines=True)

        
        ordered_filtered_commits = list(commit_df["commit_id"])
        return ordered_filtered_commits

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

        commit_diff = {}
        commit_blame = {}
        files = []
        for log in diff_log:
            files_diff = aggregator(parse_lines(log))
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
                if file_language is None or file_language.lower() not in self.languages:
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
            return {
                "commit_id": commit_id,
                "date": int(commit_date),
                "keep": 0
            }

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
        
        save_file = f"{self.raw_save_path}/{commit_id}.json"
        with open(save_file, "w") as f:
            json.dump(commit, f, indent=4)
            
        return {
                "commit_id": commit_id,
                "date": int(commit_date),
                "keep": 1
            }
    

        
       
# # Example usage
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(add_help= False)
#     parser.add_argument("--workers", type= int, default= 1, help="Number of parallel workers")
#     parser.add_argument("--language", type= str, help="Language")
#     parser.add_argument("--url", type=str, help= "Git clone url")
#     parser.add_argument("--input_path", type=str, help= "Parent directory of input repository", default= DEFAULT_INPUT)
#     parser.add_argument("--output_path", type=str, help= "Output directory", default= DEFAULT_OUTPUT)
#     parser.add_argument("--start", type=int, default=None, help= "First commit index")
#     parser.add_argument("--end", type=int, default=None, help="Last commit index")

#     params = parser.parse_args()
#     miner = Miner(params)
#     miner.run()
