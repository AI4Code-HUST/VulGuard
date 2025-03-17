import traceback
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from .features.Kamei14 import *
from .features.VCCFinder import *
from .utils.utils import *
from .Dict import Dict

def split_sentence(sentence):
    sentence = sentence.replace('.', ' . ').replace('_', ' ').replace('@', ' @ ')\
        .replace('-', ' - ').replace('~', ' ~ ').replace('%', ' % ').replace('^', ' ^ ')\
        .replace('&', ' & ').replace('*', ' * ').replace('(', ' ( ').replace(')', ' ) ')\
        .replace('+', ' + ').replace('=', ' = ').replace('{', ' { ').replace('}', ' } ')\
        .replace('|', ' | ').replace('\\', ' \ ').replace('[', ' [ ').replace(']', ' ] ')\
        .replace(':', ' : ').replace(';', ' ; ').replace(',', ' , ').replace('<', ' < ')\
        .replace('>', ' > ').replace('?', ' ? ').replace('/', ' / ')
    sentence = ' '.join(sentence.split())
    return sentence

class Extractor:
    def __init__(self, params):
        self.repo_name = None if params.repo_name is None else params.repo_name
        self.continue_run = False if params.continue_run is None else params.continue_run
        self.logger = ("logs_extractor_main.log")
        self.save_path = f"{DEFAULT_EXTRACTED_OUTPUT}/{self.repo_name}" if params.save_path is None else params.save_path 

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def run(self, path: str):
        self.file_path = path
        self.file_name = self.file_path.split('/')[-1]
        print(f"\tProcess: {self.file_name}")
        futures = []
        with ProcessPoolExecutor(max_workers=3) as executor:
            executor.submit(self.process_feature_Kamei14)
            executor.submit(self.process_feature_VCCFinder)
            executor.submit(self.process_commit)
        for future in as_completed(futures):
            pass

    def process_feature_Kamei14(self):
        try:
            features_extractor = Kamei14(self.logger)
            if self.continue_run:
                features_extractor.load_state(self.save_path)
                
            iterator = load_jsonl(self.file_path)
            for commit in tqdm(iterator, "Processing Kamei14 Features:"):
                line = [features_extractor.process(commit)]
                append_jsonl(line, f"{self.save_path}/features-{self.repo_name}.jsonl")
                if line[0]["fix"]:
                    append_jsonl(
                        [{
                            "commit_id": line[0]["commit_id"],
                            "Repository": self.repo_name
                        }], 
                        f"{self.save_path}/bugfixes-{self.repo_name}.jsonl"
                    )
            features_extractor.save_state(self.save_path)
        except Exception as e:
            print(line)
            self.logger.error(traceback.format_exc())
            exit()
            
    def process_feature_VCCFinder(self):
        try:
            features_extractor = VCCFinder(self.logger)
            if self.continue_run:
                features_extractor.load_state(self.save_path)
                
            iterator = load_jsonl(self.file_path)
            for commit in tqdm(iterator, "Processing VCCFinder Features:"):
                features_extractor.absorb(commit)
            features_extractor.save_state(self.save_path)
            features_extractor.release(f"{self.save_path}/vcc-features-{self.repo_name}.jsonl")
        except Exception as e:
            print(line)
            self.logger.error(traceback.format_exc())
            exit()

    def process_one_commit(self, commit):
        def get_std_str(string: str):
            return " ".join(split_sentence(string.strip()).split(" ")).lower()

        id = commit["commit_id"]
        message = get_std_str(commit["message"])
        added_codes = [] 
        deleted_codes = []
        patch_codes = []
        for file in commit['files']:
            added, deleted = [], []
            for hunk in commit["diff"][file]["content"]:
                patch = []
                if "ab" in hunk:
                    continue
                patch.append("added")
                if "a" in hunk:
                    for line in hunk["a"]:
                        line = get_std_str(line)
                        deleted_codes.append(line)
                        patch.append(line)
                patch.append("removed")
                if "b" in hunk:
                    for line in hunk["b"]:
                        line = get_std_str(line)
                        added_codes.append(line)
                        patch.append(line)
                patch_codes.append(" ".join(patch))
        return id, message, added_codes, deleted_codes, patch_codes

    def process_commit(self):
        msg_dict, code_dict = Dict(lower=True), Dict(lower=True)
        if self.continue_run:
            msg_dict.load_state(self.save_path)
            code_dict.load_state(self.save_path)
            
        iterator = load_jsonl(self.file_path)
        for commit in tqdm(iterator, "Processing Commits:"):
            id, message, added_codes, deleted_codes, patch_codes = self.process_one_commit(commit)
            deepjit = {
                "commit_id": id,
                "messages": message,
                "code_change": "added "+" ".join(added_codes)+" removed "+" ".join(deleted_codes)+"\n"
            }
            simcom = {
                "commit_id": id,
                "messages": message,
                "code_change": "\n".join(patch_codes)
            }
            try:
                append_jsonl([deepjit], f"{self.save_path}/deepjit-{self.repo_name}.jsonl")
                append_jsonl([simcom], f"{self.save_path}/simcom-{self.repo_name}.jsonl")

                for word in message.split():
                    msg_dict.add(word)
                for line in patch_codes:
                    for word in line.split():
                        code_dict.add(word)
                    
            except Exception as e:
                self.logger.error(e)
                self.logger.error(traceback.format_exc()) 
                exit()
                
        msg_dict.save_state(self.save_path)
        code_dict.save_state(self.save_path)
        
        pruned_msg_dict = msg_dict.prune(100000)
        pruned_code_dict = code_dict.prune(100000)
        save_jsonl([pruned_msg_dict.get_dict(), pruned_code_dict.get_dict()], f"{self.save_path}/dict-{self.repo_name}.jsonl")   