
import json
import re
from collections import defaultdict

from bs4 import BeautifulSoup
from bs4.element import Tag

from helpers import get_logger

from .get_diff import get_diff_file, get_lines_change

logger = get_logger(__name__)


def read_json_file(file_path):
    with open(file_path) as f:
        return json.load(f)


def parse_diff(commit, idx):
    # edit diff_path
    diff_path = get_diff_file(commit, idx)
    result = defaultdict(lambda: list())
    if diff_path is None:
        return result,dict()
    with open(diff_path) as f:
        content = f.read()
    if len(content) == 0:
        logger.error(
            f"Can't get diff betwen berfor and after: commit: {commit}, idx={idx}, file={diff_path}")
        return
    soup = BeautifulSoup(content, features="html.parser")
    pre_tag = soup.find("pre")
    if pre_tag is None:
        logger.error(
            f"Can't paser diff betwen berfor and after: commit: {commit}, idx={idx}, file={diff_path}")
        return
    check = False
    parser_text = pre_tag.getText()
    parser_text = parser_text[parser_text.find("@@"):]
    groups_change = defaultdict(lambda: list())
    index = 0
    for line in parser_text.splitlines():
        if line.startswith("@@"):
            index += 1
        else:
            groups_change[index].append(line)

    # add: list
    index = 0
    count_add = -1
    count_del = -1
    pre_line_add = -1
    pre_line_del = -1
    num_add = 0
    num_del = 0
    c_line = 0
    set_d = set()
    set_a = set()
    map_lines = dict()

    for child in pre_tag.children:
        text = child.getText()
        if text.startswith("@@"):
            tmp = -1
            pre_line_add = -1
            pre_line_del = -1
            assert count_add <= num_add-1
            assert count_del <= num_del-1
            count_add = -1
            count_del = -1
            c_line = 0
            index += 1
            res = re.findall("\d+", text)
            if len(res) == 2:
                c_del = int(res[0])
                c_add = int(res[1])
                num_add = 1
                num_del = 1
            elif len(res) == 3:
                c_del = int(res[0])
                if f"-{res[0]} +{res[1]}" in text:
                    c_add = int(res[1])
                    num_del = 1
                    num_add = int(res[2])
                else:
                    c_add = int(res[2])
                    num_del = int(res[1])
                    num_add = 1
            elif len(res) == 4:
                c_del = int(res[0])
                c_add = int(res[2])
                num_add = int(res[3])
                num_del = int(res[1])
                # print(res)
            tmp_num_add = num_add if num_add > 0 else 1
            tmp_num_del = num_del if num_del > 0 else 1
            map_lines[c_del+tmp_num_del] = tmp_num_add + c_add
            check = True
            continue
        else:
            tmp = 0
        if not check:
            continue
        if isinstance(child, Tag):
            lines = groups_change[index]
            if "f1" in child["class"]:
                # del
                for i, line in enumerate(lines):
                    if text in line and i + 1 >= c_line:
                        if pre_line_del != i:
                            if count_del <= num_del - 2:
                                count_del += 1
                            pre_line_del = i
                        start = line.find(text)
                        result["del"].append(
                            {"line": count_del+c_del, "text": "\n", "start_column": start, "end_column": start + len(text.strip())})
                        tmp_add = [i for i in range(c_add, c_add + num_add)]
                        if count_del + c_del not in set_d:
                            set_d.add(count_del + c_del)
                            result["map_d_to_a"].append(
                                (count_del+c_del, tmp_add))
                        lines[i] = lines[i].replace(text,'')
                        break
            if "f2" in child["class"]:
                # add
                for i, line in enumerate(lines):
                    if text in line and i + 1 >= c_line:
                        # phan them vao o line nay easy
                        if pre_line_add != i:
                            if count_add <= num_add - 2:
                                count_add += 1
                            pre_line_add = i
                        elif i > 0 and len(lines[i-1].strip()) <= 0:
                            if count_add <= num_add - 2:
                                count_add += 1
                            pre_line_add = i
                        # print(text)
                        start = line.find(text)
                        tmp_del = [i for i in range(c_del, c_del + num_del)]
                        result["add"].append(
                            {"line": count_add+c_add, "text": text.strip(), "start_column": start, "end_column": start + len(text.strip())})
                        if count_add+c_add not in set_a:
                            set_a.add(count_add+c_add)
                            result["map_a_to_d"].append(
                                (count_add+c_add, tmp_del))
                        lines[i] = lines[i].replace(text,'')
        else:
            add_line = text.count("\n") + tmp
            c_line += add_line
            tmp = 0
            if add_line > 1:
                pre_line_add
                if num_add > num_del:
                    if count_add <= num_add - 2:
                        count_add += 1
                else:
                    if count_del <= num_del - 2:
                        count_del += 1
    return result, map_lines


if __name__ == "__main__":
    pass
