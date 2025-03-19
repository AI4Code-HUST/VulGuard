from collections import defaultdict
from helpers import get_logger
from szz.core.comment_parser import parse_comments_text
from szz.core.function_parser import parse_functions_content
logger = get_logger(__name__)


class ModifierExtractor():

    def __init__(self, modified, commit_id, source_file=True):
        self.modified = modified
        if modified.nloc is None or modified.nloc > 10000:
            return
        self.diff = self.get_diff()
        self.old_path = self.modified.old_path
        self.new_path = self.modified.new_path
        self.source_file = source_file
        self.raw_code_before = self.get_raw_code_before()
        self.raw_code_after = self.get_raw_code_after()
        self.commit_id = commit_id
        self.comments_after = parse_comments_text(f"{commit_id}_after_{self.modified.filename}".replace(" ", ""), self.raw_code_after)
        self.comments_before = parse_comments_text(f"{commit_id}_before_{self.modified.filename}".replace(" ", ""), self.raw_code_before)

        self.modified_added = [el[0] for el in self.modified.diff_parsed["added"] if self.filter_comment(el[0], self.comments_after) and len(el[1].strip()) > 0]
        self.modified_deleted = [el[0] for el in self.modified.diff_parsed["deleted"] if self.filter_comment(el[0], self.comments_before) and len(el[1].strip()) > 0]

        if not source_file:
            return

        self.modified_deleted_in_function = defaultdict()
        self.modified_add_in_function = defaultdict()
        self.function_after_map = defaultdict()
        self.function_before_map = defaultdict()
        self.get_function_before()
        self.get_function_after()

    def filter_comment(self, line_num, comment_ranges):
        for comment_range in comment_ranges:
            if comment_range.start <= line_num <= comment_range.end:
                return False
        return True

    def get_raw_code_before(self):
        return self.modified.source_code_before

    def get_raw_code_after(self):
        return self.modified.source_code

    def get_function_after(self):
        if self.modified.source_code is None:
            return []
        lines_after = self.modified.source_code.splitlines()
        start_methods_after = []
        end_methods_after = []
        name_methods_after = []
        lines_in_func = defaultdict(lambda: list())
        list_funcs = list()
        # check faild of pydriller
        if self.checkParser(self.modified.methods, lines_after) or False:
            list_funcs = parse_functions_content(
                f"{self.commit_id}_after_{self.modified.filename}".replace(" ", ""), self.modified.source_code)
        else:
            for method in self.modified.methods:
                list_funcs.append(
                    {"name": method.name, "start_line": method.start_line, "end_line": method.end_line})
        names = list()
        for method in list_funcs:
            start_methods_after.append(method["start_line"]-1)
            end_methods_after.append(method["end_line"])
            name = method["name"]
            names.append(name)
            if name in names:
                name = f"{name}_{names.count(name)}"
            name_methods_after.append(name)
            self.function_after_map[name] = lines_after[method["start_line"] -
                                                        1:method["end_line"]]
        assert len(start_methods_after) == len(end_methods_after)
        assert len(start_methods_after) == len(name_methods_after)

        set_index_functions = list()

        not_in_func = list()
        for add_line in self.modified_added:
            index = -1
            for idx, start_line in enumerate(start_methods_after):
                if start_line <= add_line and end_methods_after[idx] >= add_line:
                    index = idx
                    break
            if index != -1:
                assert add_line-start_line >= 0
                lines_in_func[index].append(add_line-start_line)
                if index not in set_index_functions:
                    set_index_functions.append(index)
            else:
                not_in_func.append(lines_after[add_line-1])
        for idx in set_index_functions:
            lines = ",".join([str(x) for x in lines_in_func[idx]])
            function_after = lines_after[start_methods_after[idx]: end_methods_after[idx]]
            self.modified_add_in_function[name_methods_after[idx]] = {
                "line": lines, "raw": "\n".join(function_after), "start_function": start_methods_after[idx], "end_function": end_methods_after[idx]}

    def get_function_before(self):
        if self.modified.source_code_before is None:
            return []
        lines_before = self.modified.source_code_before.splitlines()
        start_methods_before = []
        end_methods_before = []
        name_methods_before = []
        lines_in_func = defaultdict(lambda: list())
        ext = self.modified.filename.rsplit('.', 1)
        list_funcs = list()
        if self.checkParser(self.modified.methods_before, lines_before) or False:
            list_funcs = parse_functions_content(f"{self.commit_id}_before_{self.modified.filename}".replace(
                " ", ""), self.modified.source_code_before)
        else:
            for method in self.modified.methods_before:
                list_funcs.append(
                    {"name": method.name, "start_line": method.start_line, "end_line": method.end_line})

        names = list()
        for method in list_funcs:
            start_methods_before.append(method["start_line"]-1)
            end_methods_before.append(method["end_line"])
            name = method["name"]
            names.append(name)
            if name in names:
                name = f"{name}_{names.count(name)}"
            name_methods_before.append(name)
            self.function_before_map[name] = lines_before[method["start_line"] -
                                                          1:method["end_line"]]
        assert len(start_methods_before) == len(end_methods_before)
        assert len(start_methods_before) == len(name_methods_before)

        set_index_functions = list()
        not_in_func = list()
        for delete_line in self.modified_deleted:
            index = -1
            for idx, start_line in enumerate(start_methods_before):
                if start_line <= delete_line and end_methods_before[idx] >= delete_line:
                    index = idx
                    break
            if index != -1:
                assert delete_line-start_line >= 0
                lines_in_func[index].append(delete_line-start_line)
                if index not in set_index_functions:
                    set_index_functions.append(index)
            else:
                not_in_func.append(lines_before[delete_line-1])

        for idx in set_index_functions:
            lines = ",".join([str(x) for x in lines_in_func[idx]])
            function_before = lines_before[start_methods_before[idx]: end_methods_before[idx]]
            self.modified_deleted_in_function[name_methods_before[idx]] = {
                "line": lines, "raw": "\n".join(function_before), "start_function": start_methods_before[idx], "end_function": end_methods_before[idx]}

    def get_diff(self):
        return self.modified.diff

    def checkParser(self, methods, source_lines):
        if len(methods) <= 0:
            return True
        if methods[-1].end_line / len(source_lines) < 0.9:
            return True
        if len(source_lines) - methods[-1].end_line > 1000:
            return True
        return False
