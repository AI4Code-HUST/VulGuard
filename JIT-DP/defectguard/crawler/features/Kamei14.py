import math, re
import logging
import traceback
import numpy as np
from typing import Dict, List, Tuple, Set

from ..utils.utils import save_json, load_json, DEFAULT_EXTRACTED_OUTPUT

STRONG_VUL = re.compile(r'(?i)(denial.of.service|remote.code.execution|\bopen.redirect|OSVDB|\bXSS\b|\bReDoS\b|\bNVD\b|malicious|x−frame−options|attack|cross.site|exploit|directory.traversal|\bRCE\b|\bdos\b|\bXSRF\b|clickjack|session.fixation|hijack|advisory|insecure|security|\bcross−origin\b|unauthori[z|s]ed|infinite.loop)')
MEDIUM_VUL =re.compile(r'(?i)(authenticat(e|ion)|bruteforce|bypass|constant.time|crack|credential|\bDoS\b|expos(e|ing)|hack|harden|injection|lockout|overflow|password|\bPoC\b|proof.of.concept|poison|privelage|\b(in)?secur(e|ity)|(de)?serializ|spoof|timing|traversal)')

class Kamei14:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

        self.keep_track_authors = {}
        """
        {
            author_name: {
                file_1: [mod_date_1, mod_date_2, ...]
                file_2: [mod_date_1, mod_date_2, ...]
            }
        }
        """

        self.keep_track_files = {}
        """
        {
            file_name: {
                "author": list(author name)
                "unique changes": int
                "last modified date": int(datetime) 
            }
        }
        """
        
    def process(self, commit: Dict) -> Dict:
        try:
            self.update(commit)
        except Exception as e:
            self.logger.error(e)
            self.logger.error(traceback.format_exc())
            exit()
        
        # self.logger.info(self.keep_track_authors)
        fix = self.is_fixing_commit(commit["message"])
        la, ld, lt, entropy = self.line_features(commit["diff"])
        ns, nd, nf = self.directory_features(commit["files"])
        ndev, nuc, age = self.file_features(commit["files"])
        exp, rexp, sexp = self.experience_features(commit["author"], commit["date"], ns)

        return {
            "commit_id": commit["commit_id"],
            "date": commit["date"],
            "ns": len(ns),
            "nd": len(nd),
            "nf": len(nf),
            "entropy": entropy,
            "la": la,
            "ld": ld,
            "lt": lt,
            "fix": fix,
            "ndev": ndev,
            "age": age,
            "nuc": nuc,
            "exp": exp,
            "rexp": rexp,
            "sexp": sexp,
        }

    def update(self, commit: Dict):
        author_record = self.keep_track_authors.get(commit["author"], None)
        if author_record is None:
            self.keep_track_authors[commit["author"]] = {}
            author_record = self.keep_track_authors[commit["author"]]

        for file_name in commit["files"]:
            file_record = self.keep_track_files.get(file_name, None)
            if file_record is None:
                self.keep_track_files[file_name] = {
                    "authors": [commit["author"]],
                    "unique changes": 1,
                    "last modified date": commit["date"],
                    "time interval": 0
                }
            else:
                file_record["authors"].append(commit["author"])
                file_record["authors"] = list(set(file_record["authors"]))
                file_record["unique changes"] += 1
                file_record["time interval"] = commit["date"] - file_record["last modified date"]
                file_record["last modified date"] = commit["date"]

            author_file_record = author_record.get(file_name, None)
            if author_file_record is None:
                author_record[file_name] = [commit["date"]]
            else:
                author_file_record.append(commit["date"])

    def line_features (self, diff: Dict) -> Tuple[int, int, int, int]:
        la, ld, lt = 0, 0, 0
        try:
            each_file = []
            for file_name, file_diff in diff.items():
                temp = [0, 0]
                for chunk in file_diff["content"]:
                    added = chunk.get("a", [])
                    temp[0] += len(added)

                    deleted = chunk.get("b", [])
                    temp[1] += len(deleted)
                each_file.append(temp[0] + temp[1])
                la += temp[0]
                ld += temp[1]
                lt = file_diff["meta_a"]["lines"] if "meta_a" in file_diff else 0
        except Exception as e:
            print(e)

        entropy = 0
        totalLOCModified = np.sum(each_file)
        for fileLocMod in each_file:
            if (fileLocMod != 0 ):
                avg = fileLocMod/totalLOCModified
                entropy -= ( avg * math.log( avg,2 ) )
        
        return la, ld, lt, entropy

    def directory_features (self, file_paths: List) -> Tuple[Set[str], Set[str], Set[str]]:
        ns, nd, nf = set(), set(), set()
        for file_path in file_paths:
            fileDirs = file_path.split("/")
            subsystem = "root" if len(fileDirs) == 1 else fileDirs[0]
            directory = "root" if len(fileDirs) == 1 else "/".join(fileDirs[0:-1])
            file_name = fileDirs[-1]

            ns.add(subsystem)
            nd.add(directory)
            nf.add(file_name)

        self.logger.info(ns)
        self.logger.info(nd)
        self.logger.info(nf)
        return ns, nd, nf

    def is_fixing_commit(self, file_message: str) -> int:
        m = STRONG_VUL.search(file_message)
        n = MEDIUM_VUL.search(file_message)
        return 1 if m or n else 0 
        # bug_keywords = ["fix", "bug", "issue"]
        # wrong_keywords = ["fix typo", "fix build", "non-fix"]
        # if any(keyword in file_message for keyword in bug_keywords):
        #     if not any(keyword in file_message for keyword in wrong_keywords):
        #         return 1
        # return 0
    
    def file_features (self, file_names: List[str]) -> Tuple[int, int, int]:
        ndev, nuc, ages = set(), 0, []
        for file_name in file_names:
            file_record = self.keep_track_files[file_name]
            assert file_record is not None
            
            ndev = len(file_record["authors"])
            nuc += file_record["unique changes"]
            ages.append(file_record["time interval"])

        age = np.mean(ages) / 86400 if ages else 0
        return ndev, nuc, age

    def experience_features (self, author: str, anchor_date: int, ns: Set[str], ) -> Tuple[int, int, int]:
        author_record = self.keep_track_authors[author]
        assert author_record is not None
        self.logger.info(ns)
        exp, rexp, sexp = 0, 0, 0
        for file_name, commit_dates in author_record.items():
            dirs = file_name.split("/")
            sub = "root" if len(dirs) == 1 else dirs[0]
            self.logger.info(sub)
            exp += len(commit_dates)
            rexp += np.sum([1 / (max ( (anchor_date - date) / 86400, 0 ) + 1) for date in commit_dates])
            sexp += 1 if sub in ns else 0

        return exp, rexp, sexp
    
    def save_state(self, path: str) -> None:
        save_json([self.keep_track_authors, self.keep_track_files], f"{path}/Kamei14_state_dict.json")

    def load_state(self, path: str) -> None:
        try:
            self.keep_track_authors, self.keep_track_files = load_json(f"{path}/Kamei14_state_dict.json")
        except FileNotFoundError as e:
            self.logger.error(f"{path} : {e}")