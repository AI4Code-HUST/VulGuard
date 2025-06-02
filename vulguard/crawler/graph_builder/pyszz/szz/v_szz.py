import sys
import traceback
from collections import defaultdict
from typing import List

from pydriller import ModificationType, Git as PyDrillerGitRepo

from helpers import compute_line_ratio
from szz.core.abstract_szz import AbstractSZZ, ImpactedFile

MAXSIZE = sys.maxsize


class V_SZZ(AbstractSZZ):
    """
    My SZZ implementation.

    Supported **kwargs:

    * ignore_revs_file_path

    """

    def __init__(self, repo_name: str, repo_dir: str):
        super().__init__(repo_name, repo_dir)

    def find_vcc(self, fix_commit_hash: str, impacted_files: List['ImpactedFile'], **kwargs):
        """
        Find bug contributing commit candidates.

        :param str fix_commit_hash: hash of fix commit to scan for buggy commits
        :param List[ImpactedFile] impacted_files: list of impacted files in fix commit
        :key ignore_revs_file_path (str): specify ignore revs file for git blame to ignore specific commits.
        :returns Set[Commit] a set of bug contributing commit candidates, represented by Commit object
        """

        # log.info(f"find_vcc() kwargs: {kwargs}")

        ignore_revs_file_path = kwargs.get('ignore_revs_file_path', None)
        # self._set_working_tree_to_commit(fix_commit_hash)

        bug_introd_commit_dict = defaultdict(dict)
        for imp_file in impacted_files:
            # print('impacted file', imp_file.file_path)
            try:
                blame_data = self._blame(
                    # rev='HEAD^',
                    rev='{commit_id}^'.format(commit_id=fix_commit_hash),
                    file_path=imp_file.file_path,
                    modified_lines=imp_file.modified_lines,
                    ignore_revs_file_path=ignore_revs_file_path,
                    ignore_whitespaces=True,
                    skip_comments=True
                )

                for entry in blame_data:
                    commit_id = entry.commit.hexsha
                    if commit_id not in bug_introd_commit_dict:
                        bug_introd_commit_dict[commit_id] = {'commit_hash': entry.commit.hexsha,
                                                             "commit_timestamp": entry.commit.committed_date,
                                                             "vulnerable_changes": defaultdict(list)}
                    bug_introd_commit_dict[commit_id]["vulnerable_changes"][entry.file_path].append(entry.line_num)
                    # bug_introd_commits.append(previous_commits)
            except:
                print(traceback.format_exc())
        bug_introd_commits = list(bug_introd_commit_dict.values())
        for c in bug_introd_commits:
            vul_changes_info = c["vulnerable_changes"]
            for file_path in vul_changes_info:
                vul_changes_info[file_path].sort()
        if len(bug_introd_commits) > 1:
            bug_introd_commits.sort(key=lambda cmt: cmt["commit_timestamp"], reverse=True)
        return bug_introd_commits

    def map_modified_line(self, blame_entry, blame_file_path):
        # TODO: rename type
        blame_commit = PyDrillerGitRepo(self.repository_path).get_commit(blame_entry.commit.hexsha)
        # print('get blame commit', blame_commit, blame_entry.commit.hexsha)

        for mod in blame_commit.modified_files:
            file_path = mod.new_path
            if mod.change_type == ModificationType.DELETE or mod.change_type == ModificationType.RENAME:
                file_path = mod.old_path

            if file_path != blame_file_path:
                continue

            if not mod.old_path:
                # "newly added"
                return -1

            lines_added = [added for added in mod.diff_parsed['added']]
            lines_deleted = [deleted for deleted in mod.diff_parsed['deleted']]

            if len(lines_deleted) == 0:
                return -1

            # print('line added/deleted', len(lines_added), len(lines_deleted))

            if blame_entry.line_str:
                sorted_lines_deleted = [(line[0], line[1],
                                         compute_line_ratio(blame_entry.line_str, line[1]),
                                         abs(blame_entry.line_num - line[0]))
                                        for line in lines_deleted]
                sorted_lines_deleted = sorted(sorted_lines_deleted, key=lambda x: (x[2], MAXSIZE - x[3]), reverse=True)
                # print(sorted_lines_deleted)

                # print(sorted_lines_deleted)
                if sorted_lines_deleted[0][2] > 0.75:
                    return sorted_lines_deleted[0][0]

        return -1


def merge_impacted_files(imp_files_1, imp_files_2):
    imp_files_dict_1 = {imp_f.file_path: imp_f.modified_lines for imp_f in imp_files_1}
    imp_files_dict_2 = {imp_f.file_path: imp_f.modified_lines for imp_f in imp_files_2}
    merged_imp_files = []
    for file_path in set(list(imp_files_dict_1.keys()) + list(imp_files_dict_2.keys())):
        mod_lines = list(set(imp_files_dict_1.get(file_path, []) + imp_files_dict_2.get(file_path, [])))
        merged_imp_files.append(ImpactedFile(file_path, mod_lines))
    return merged_imp_files