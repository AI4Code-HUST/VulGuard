import bisect
import sys

from pydriller import ModifiedFile

from config import BASE_DIR
from file_manager import get_file_name, write_file
from helpers import subprocess_cmd, get_logger, compute_line_ratio

logger = get_logger(__name__)


def write_unchanged_line_mapping_after_diff(file_path_1, file_path_2, output_diff_mapping_file_path):
    logger.info(f"Writing unchanged line mapping from [{get_file_name(file_path_1)} and {get_file_name(file_path_2)}]")
    command = " ".join(["diff",
                        "--changed-group-format=",
                        "--unchanged-group-format='%df %dl %dF %dL,'",
                        file_path_1,
                        file_path_2])

    stdout, stderr = subprocess_cmd(command)
    if stderr:
        logger.warning(f"[{file_path_1}->{file_path_2}]{stderr}")
        return {}
    output = stdout

    mapping_output_dict = {}
    for item in output.split(",")[:-1]:
        start1, end1, start2, end2 = [int(s) for s in item.split()]
        n = end1 - start1 + 1
        assert (n == end2 - start2 + 1)  # unchanged group, should be same length in each file
        for i in range(n):
            mapping_output_dict[start1 + i] = start2 + i

    mapping_output_str = "\n".join([f"{k}:{v}" for k, v in mapping_output_dict.items()])
    mapping_file_content = f"{file_path_2.replace(BASE_DIR, '')}->{file_path_1.replace(BASE_DIR, '')}\n{mapping_output_str}"
    write_file(output_diff_mapping_file_path, mapping_file_content)


def read_mapping_line_file_path(diff_line_mapping_file_path):
    mapping_lines = [l.strip() for l in open(diff_line_mapping_file_path).readlines()][1:]
    mapping_line_dict = {}
    for l in mapping_lines:
        new_l, old_l = l.split(":")
        mapping_line_dict[int(new_l)] = int(old_l)
    return mapping_line_dict


def find_modified_lines_in_commit_changes(mod_file: ModifiedFile, diff_line_mapping_file_path: str):
    mapping_line_dict = read_mapping_line_file_path(diff_line_mapping_file_path)

    commit_deleted_line_dict = {deleted[0]: deleted[1] for deleted in mod_file.diff_parsed["deleted"]}
    commit_deleted_line_nums = sorted(commit_deleted_line_dict.keys())
    old_file_lines = mod_file.source_code_before.splitlines()

    commit_added_line_dict = {added[0]: added[1] for added in mod_file.diff_parsed["added"]}
    commit_added_line_nums = sorted(commit_added_line_dict.keys())
    new_file_lines = mod_file.source_code.splitlines()

    assert len(old_file_lines) >= len(mapping_line_dict.values()) and len(new_file_lines) >= len(
        mapping_line_dict.keys())

    mapping_line_dict[0] = 0
    mapping_line_dict[len(new_file_lines) + 1] = len(old_file_lines) + 1

    unchanged_lines_in_after_file = sorted(mapping_line_dict.keys())

    # from each added line in [after] changed file, find deleted lines in close location from [before] changed
    # then measure similarity to check if the operation is (1) modified existing line or (2) added a complete new line
    after_modified_line_nums = set()
    for added_ln in commit_added_line_nums:
        insertion_index = bisect.bisect_left(unchanged_lines_in_after_file, added_ln)
        before_added_ln = unchanged_lines_in_after_file[insertion_index - 1]
        after_added_ln = unchanged_lines_in_after_file[insertion_index]
        before_deleted_line = mapping_line_dict[before_added_ln]
        after_deleted_line = mapping_line_dict[after_added_ln]
        sorted_lines_deleted = [(deleted_ln, added_ln, compute_line_ratio(commit_deleted_line_dict[deleted_ln],
                                                                          commit_added_line_dict[added_ln]),
                                 abs(added_ln - deleted_ln)) for deleted_ln in commit_deleted_line_nums if
                                before_deleted_line < deleted_ln < after_deleted_line
                                and deleted_ln not in after_modified_line_nums]
        if len(sorted_lines_deleted) <= 0:
            continue
        sorted_lines_deleted = sorted(sorted_lines_deleted, key=lambda x: (x[2], -x[3]), reverse=True)
        if sorted_lines_deleted[0][2] > 0.75:
            after_modified_line_nums.add(sorted_lines_deleted[0][1])

    return after_modified_line_nums
