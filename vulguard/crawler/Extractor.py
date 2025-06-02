import traceback
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import List

from .features.Kamei14 import *
from .features.VCCFinder import *
from .utils.utils import *
from .Dict import Dictionary

class Extractor:
    def __init__(self, params):
        self.repo_name = params.repo_name
        self.save_path = f"{params.dataset_save_path}"
        self.raw_path = f"{params.dataset_save_path}/raw"
        self.logger = params.logger
        
        os.makedirs(f"{self.save_path}/extracted", exist_ok=True)
    
    def run(self, commit_ids: List):
        futures = []        
        with ProcessPoolExecutor(max_workers=3) as executor:
            # Process extracting code changes and messages
            executor.submit(self.process_commits, commit_ids)
            
            # Processes extracting features, add your own features here
            executor.submit(self.process_feature_Kamei14, commit_ids)
            executor.submit(self.process_feature_VCCFinder, commit_ids)
        
        for future in as_completed(futures):
            pass
    
    def iter_commits(self, commit_ids: List):
        for id in commit_ids:
            raw_file = f"{self.raw_path}/{id}.json"
            with open(raw_file, "r") as f:
                commit = json.load(f)
            yield commit

    def process_feature_Kamei14(self, commit_ids: List):
        try:
            features_extractor = Kamei14(self.logger)
            commits = self.iter_commits(commit_ids)
            
            lines = []
            for commit in tqdm(commits, "Processing Kamei14 Features:"):
                line = features_extractor.process(commit)
                lines.append(line)
                
            save_jsonl(lines, f"{self.save_path}/extracted/Kamei_features_{self.repo_name}.jsonl")
            features_extractor.save_state(f"{self.save_path}/extracted")
        except Exception as e:
            self.logger.error(traceback.format_exc())
            exit()
            
    def process_feature_VCCFinder(self, commit_ids):
        try:
            features_extractor = VCCFinder(self.logger)
            
            commits = self.iter_commits(commit_ids)
            for commit in tqdm(commits, "Processing VCCFinder Features:"):
                features_extractor.absorb(commit)
            
            features_extractor.save_state(f"{self.save_path}/extracted")
            features_extractor.release(f"{self.save_path}/extracted/VCC_features_{self.repo_name}.jsonl")
        except Exception as e:
            self.logger.error(traceback.format_exc())
            exit()

    def process_one_commit(self, commit):
        def get_std_str(string: str):
            return " ".join(split_sentence(string.strip()).split(" ")).lower()

        id = commit["commit_id"]
        message = get_std_str(commit["message"])
        added_codes, deleted_codes, patch_codes = [], [], []
        
        for file in commit['files']:
            for hunk in commit["diff"][file]["content"]:
                patch = []
                if "ab" in hunk:
                    continue
                patch.append("<ADD>")
                if "a" in hunk:
                    for line in hunk["a"]:
                        line = get_std_str(line)
                        deleted_codes.append(line)
                        patch.append(line)
                patch.append("<REMOVE>")
                if "b" in hunk:
                    for line in hunk["b"]:
                        line = get_std_str(line)
                        added_codes.append(line)
                        patch.append(line)
                patch_codes.append(" ".join(patch))
                
        
        return id, message, added_codes, deleted_codes, patch_codes

    def process_commits(self, commit_ids: List):
        msg_dict, code_dict = Dictionary(lower=True), Dictionary(lower=True)
        
        deepjit, simcom, vfc = [], [], []
        
        commits = self.iter_commits(commit_ids)
        for commit in tqdm(commits, "Processing Commits:"):
            id, message, added_codes, deleted_codes, patch_codes = self.process_one_commit(commit)
            
            deepjit.append ( {
                "commit_id": id,
                "messages": message,
                "code_change": "<ADD> " + " ".join(added_codes) + " <REMOVE> " + " ".join(deleted_codes)+"\n"
            } )
            
            simcom.append ( {
                "commit_id": id,
                "messages": message,
                "code_change": "\n".join(patch_codes)
            } )

            if is_vfc(message):
                vfc.append ( {
                    "commit_id": id,
                    "Repository": self.repo_name   
                } )

            for word in message.split():
                msg_dict.add(word)
            for line in patch_codes:
                for word in line.split():
                    code_dict.add(word)
                    
                    
        # msg_dict.save_state(self.save_path)
        # code_dict.save_state(self.save_path)
        
        pruned_msg_dict = msg_dict.prune(100000)
        pruned_code_dict = code_dict.prune(100000)
        
        
        save_jsonl(deepjit, f"{self.save_path}/extracted/merge_{self.repo_name}.jsonl")
        save_jsonl(simcom, f"{self.save_path}/extracted/patch_{self.repo_name}.jsonl")
        save_jsonl(vfc, f"{self.save_path}/vfc_{self.repo_name}.jsonl")
        save_jsonl([pruned_msg_dict.get_dict(), pruned_code_dict.get_dict()], f"{self.save_path}/dict_{self.repo_name}.jsonl")   