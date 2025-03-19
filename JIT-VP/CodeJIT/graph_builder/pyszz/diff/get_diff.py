from collections import defaultdict

from config import JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR
from file_manager import get_absolute_path, is_path_exist, join_path
from helpers import get_logger, subprocess_cmd

logger = get_logger(__name__)
SCRIPT_PATH = "pyszz/diff/script/ansi2html.sh"


def get_diff_file(commit, idx):
    path = join_path(JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR, commit)
    b_path = join_path(path, f"before.{idx}.cpp")
    a_path = join_path(path, f"after.{idx}.cpp")
    o_path = join_path(path, f"diff.{idx}.html")
    # if is_path_exist(o_path):
    #     return o_path

    cmd = f"git diff --no-index --word-diff=color -U0 {b_path} {a_path} | bash {get_absolute_path(SCRIPT_PATH)} > {o_path}"
    print(f"Get diff {commit}")
    rs = subprocess_cmd(cmd)
    if len(rs[1]) == 0:
        return o_path
    logger.error(f"Can't get diff {commit}, {idx}, {rs[1]}")
    return None


def get_diff_text(commit, idx):
    path = join_path(JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR, commit)
    b_path = join_path(path, f"before.{idx}.cpp")
    a_path = join_path(path, f"after.{idx}.cpp")
    cmd = f"git diff --no-index --no-prefix -U1000 {b_path} {a_path}"
    rs = subprocess_cmd(cmd)
    if len(rs[1]) == 0:
        return rs[0]
    logger.error(f"Can't get diff {commit}, {idx}, {rs[1]}")
    return None


def get_lines_change(commit, idx):
    diff_text = get_diff_text(commit, idx)
    result = defaultdict(lambda: list())
    line_number_add = 0
    line_number_del = 0
    check = False
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            check = True
            continue
        if not check:
            continue
        line_number_del += 1
        line_number_add += 1

        if line.startswith("+"):
            result["add"].append(line_number_add)
            line_number_del -= 1
        if line.startswith("-"):
            result["del"].append(line_number_del)
            line_number_add -= 1
    return result
