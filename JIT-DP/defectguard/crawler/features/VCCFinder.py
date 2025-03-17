import math, re, os, shutil
import logging
import traceback
import numpy as np
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
from ..utils.utils import save_json, load_json, append_jsonl, load_jsonl, create_console_log_handler

keywords_62 = [
    "do", "if", "asm", "for", "int", "new", "try", "auto", "bool", "case", "char", "else", "enum", "free", "goto", "long", 
    "this", "true", "void", "alloc", "break", "catch", "class", "const", "false", "float", "short", "throw", "union", "using", 
    "while", "alloca", "calloc", "delete", "double", "extern", "friend", "inline", "malloc", "public", "return", "signed", 
    "sizeof", "static", "struct", "switch", "typeid", "default", "mutable", "private", "realloc", "typedef", "virtual", "wchar_t", 
    "continue", "explicit", "operator", "register", "template", "typename", "unsigned", "volatile", "namespace", "protected", 
    "const_cast", "static_cast", "dynamic_cast", "reinterpret_cast"
]

class VCCFinder:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
        self.keep_track_meta = {
            "commits": {},
            "authors": {},
            "files": {},
            "total_commit": 0, 
            "total_contributors": 0 
        }
    
    def absorb(self, commit: Dict) -> None:
        try:
            commit_id, author, files, diff = commit["commit_id"], commit["author"], commit["files"], commit["diff"]
            addition, deletion, hunk_count = self.code_metrics(diff)
            self.keep_track_meta["commits"][commit_id] = {
                "author" : author,
                "files": files,
                "addition": addition,
                "deletion": deletion,
                "hunk_count": hunk_count
            }
            
            patch_keywords = self.patch_keywords(diff)
            self.keep_track_meta["commits"][commit_id].update(patch_keywords)
            
            author_contribution = self.keep_track_meta["authors"].get(author, [])
            author_contribution.append(commit_id)
            self.keep_track_meta["authors"][author] = author_contribution
            
            for file in files:
                file_tracker = self.keep_track_meta["files"].get(file, {"commit_id": [], "author": []})
                file_tracker["commit_id"].append(commit_id)
                if author not in file_tracker["author"]:
                    file_tracker["author"].append(author)
                self.keep_track_meta["files"][file] = file_tracker
                
            self.keep_track_meta["total_commit"] += 1
            self.keep_track_meta["total_contributors"] = len(self.keep_track_meta["authors"])
            
        except Exception as e:
            self.logger.error(f"absorb {e}")
            self.logger.error(traceback.format_exc())
            exit()
    
    def release (self, file: str):
        if os.path.exists(file):
            shutil.rmtree(file)
          
        number_unique_contributors = self.keep_track_meta["total_contributors"]    
        feat = {}
        for commit_id in tqdm(self.keep_track_meta["commits"]):
            author = self.keep_track_meta["commits"][commit_id]["author"]
            files = self.keep_track_meta["commits"][commit_id]["files"]
            
            past_changes, future_changes, past_different_authors, future_different_authors = self.get_changes(commit_id, author, files)
            author_contributions_percent = self.get_author_contributions_percent(author)      
            
            feat["commit_id"] = commit_id
            feat.update(self.keep_track_meta["commits"][commit_id])
            feat["author_contributions_percent"] = author_contributions_percent
            feat["past_changes"] = past_changes
            feat["future_changes"] = future_changes
            feat["past_different_authors"] = past_different_authors
            feat["future_different_authors"] = future_different_authors
            append_jsonl([feat], file)
                        
    def code_metrics (self, diff: Dict) -> Tuple[int, int, int]:
        addition, deletion, hunk_count = 0, 0, 0
        try:
            for file_name, file_diff in diff.items():
                for chunk in file_diff["content"]:
                    added = chunk.get("a", [])
                    addition += len(added)

                    deleted = chunk.get("b", [])
                    deletion += len(deleted)
                    
                    hunk_count += 1 if "ab" in chunk else 0
                
        except Exception as e:
            self.logger.error(f"code {e}")
            exit()
            
        return addition, deletion, hunk_count
    
    def patch_keywords (self, diff: Dict) -> Dict:
        kw_map = {f"kw_{key}" : 0 for key in keywords_62}
        try:
            def count_kw(chunk):
                for line in chunk:
                    for word in line.split():
                        if word in keywords_62:
                            kw_map[f"kw_{word}"] += 1
                    
            for file_name, file_diff in diff.items():
                for chunk in file_diff["content"]:
                    if "ab" in chunk:
                        continue
                    count_kw(chunk.get("a", []))
                    count_kw(chunk.get("b", []))
                    
        except Exception as e:
            self.logger.error(f"patch {e}")
            exit()   
        
        return kw_map
    
    def get_changes (self, commit_id: str, author: str, files: List[str]) -> Tuple[int, int, int, int]:
        past_changes, future_changes, past_different_authors, future_different_authors = 0, 0, 0, 0
        for file in files:
            total_file_change = len(self.keep_track_meta["files"][file]["commit_id"])
            commit_index = self.keep_track_meta["files"][file]["commit_id"].index(commit_id)
            past_changes += commit_index
            future_changes += total_file_change - commit_index
            
            total_author = len(self.keep_track_meta["files"][file]["author"])
            author_index = self.keep_track_meta["files"][file]["author"].index(author)
            past_different_authors += author_index
            future_different_authors += total_author - author_index
        return past_changes, future_changes, past_different_authors, future_different_authors
            
    def get_author_contributions_percent (self, author: str) -> float:
        author_contributions = len(self.keep_track_meta["authors"].get(author, []))
        total_contributions = self.keep_track_meta["total_commit"]
        author_contributions_percent = author_contributions / total_contributions
        return author_contributions_percent
    
    
    def save_state(self, path: str) -> None:
        save_json(self.keep_track_meta, f"{path}/VCCFeat_state_dict.json")

    def load_state(self, path: str) -> None:
        try:
            self.keep_track_meta = load_json(f"{path}/VCCFeat_state_dict.json")
        except FileNotFoundError as e:
            self.logger.error(f"{path} : {e}")
            exit()
            
def find_files(regex_pattern: str, folder: str) -> List[str]:
    pattern = re.compile(regex_pattern)
    matching_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if pattern.match(file):
                matching_files.append(os.path.join(root, file))
    return matching_files

if __name__ == "__main__":
    path = r"E:\ALL_DATA\raw_data\FFmpeg"
    file_regex = r"^extracted-all-FFmpeg-start-\d+-end-\d+.jsonl$"
    
    file_paths = find_files(file_regex, path)
    logger = create_console_log_handler("vcc")
    features_extractor = VCCFinder(logger)

    for file in tqdm(file_paths):
        with tqdm() as bar:
            for commit in load_jsonl(file):
                features_extractor.absorb(commit)
                bar.update(1)
        features_extractor.save_state("E:/NewCrawler/vcc")
    features_extractor.load_state("E:/NewCrawler/vcc")
    features_extractor.release("E:/NewCrawler/vcc/vcc-features-extracted-all-FFmpeg.jsonl")