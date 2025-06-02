import logging as log
import re
from collections import namedtuple

from config import C_FILE_EXTENSIONS, SOURCE_CODE_OUTPUTS_DIR
from file_manager import get_file_name_with_parent, write_file, join_path, is_path_exist
from helpers import subprocess_cmd

CommentRange = namedtuple('CommentRange', 'start end')
# srcml_file_ext = ['.c', '.h', '.hh', '.hpp', '.hxx', '.cxx', '.cpp', '.cc', '.cs', '.java']

__CACHE_LINE_COMMENT_RANGES = {}


def parse_comments(source_file_path: str):
    if source_file_path not in __CACHE_LINE_COMMENT_RANGES:
        line_comment_ranges = parse_comments_srcml(source_file_path)
        __CACHE_LINE_COMMENT_RANGES[source_file_path] = line_comment_ranges
    line_comment_ranges = __CACHE_LINE_COMMENT_RANGES[source_file_path]
    return line_comment_ranges


def parse_comments_srcml(source_file_path):
    line_comment_ranges = list()

    if any(source_file_path.endswith(e) for e in C_FILE_EXTENSIONS):
        stdout, stderr = subprocess_cmd(f'srcml --position {source_file_path}')
        if not stderr:
            stdout, stderr = subprocess_cmd(f'srcml --position {source_file_path}')
            for i, line in enumerate(stdout.splitlines()):
                if i == 1 and "<comment " in line.strip() or line.strip().startswith("<comment "):
                    line_comment_ranges.append(CommentRange(start=int(re.search('pos:start="(\d+):', line).groups()[0]),
                                                            end=int(re.search('pos:end="(\d+):', line).groups()[0])))
        else:
            log.warning(f"Error while parsing file {source_file_path}")
    else:
        log.warning(f"File not supported by srcML: {get_file_name_with_parent(source_file_path)}")

    return line_comment_ranges


def js_comment_parser(file_str, file_name):
    line_comment_ranges = list()

    if file_name.endswith(".js"):
        lines = file_str.splitlines()
        l_idx = 0
        while l_idx < len(lines):
            line = lines[l_idx].strip()
            if line.startswith("/*"):
                for i in range(l_idx, len(lines)):
                    line = lines[i].strip()
                    if i == l_idx:
                        line = line[2:]
                    if line and line.endswith("*/"):
                        line_comment_ranges.append(CommentRange(start=(l_idx + 1), end=(i + 1)))
                        l_idx = i
                        break
            elif line.startswith("//"):
                line_comment_ranges.append(CommentRange(start=(l_idx + 1), end=(l_idx + 1)))
            l_idx += 1
    else:
        log.error(f"unable to parse comments for: {file_name}")

    return line_comment_ranges


def php_comment_parser(file_str, file_name):
    line_comment_ranges = list()

    if file_name.endswith(".php"):
        lines = file_str.splitlines()
        l_idx = 0
        while l_idx < len(lines):
            line = lines[l_idx].strip()
            if line.startswith("/*"):
                for i in range(l_idx, len(lines)):
                    line = lines[i].strip()
                    if i == l_idx:
                        line = line[2:]
                    if line and line.endswith("*/"):
                        line_comment_ranges.append(CommentRange(start=(l_idx + 1), end=(i + 1)))
                        l_idx = i
                        break
            elif line.startswith("//") or line.startswith("#"):
                line_comment_ranges.append(CommentRange(start=(l_idx + 1), end=(l_idx + 1)))
            l_idx += 1
    else:
        log.error(f"unable to parse comments for: {file_name}")

    return line_comment_ranges


def rb_comment_parser(file_str, file_name):
    line_comment_ranges = list()

    if file_name.endswith(".rb"):
        lines = file_str.splitlines()
        l_idx = 0
        while l_idx < len(lines):
            line = lines[l_idx].strip()
            if line.startswith("=begin"):
                for i in range(l_idx, len(lines)):
                    line = lines[i].strip()
                    if line and line.endswith("=end"):
                        line_comment_ranges.append(CommentRange(start=(l_idx + 1), end=(i + 1)))
                        l_idx = i
                        break
            elif line.startswith("//") or line.startswith("#"):
                line_comment_ranges.append(CommentRange(start=(l_idx + 1), end=(l_idx + 1)))
            l_idx += 1
    else:
        log.error(f"unable to parse comments for: {file_name}")

    return line_comment_ranges


def py_comment_parser(file_str, file_name):
    line_comment_ranges = list()

    if file_name.endswith(".py"):
        lines = file_str.splitlines()
        l_idx = 0
        while l_idx < len(lines):
            line = lines[l_idx].strip()
            if line.startswith("'''") or line.startswith('"""'):
                for i in range(l_idx, len(lines)):
                    line = lines[i].strip()
                    if i == l_idx:
                        line = line[3:]
                    if line and (
                            line.endswith("'''") or line.endswith('"""') or line.startswith("'''") or line.startswith(
                            '"""')):
                        line_comment_ranges.append(CommentRange(start=(l_idx + 1), end=(i + 1)))
                        l_idx = i
                        break
            elif line.startswith("#"):
                line_comment_ranges.append(CommentRange(start=(l_idx + 1), end=(l_idx + 1)))
            l_idx += 1
    else:
        log.error(f"unable to parse comments for: {file_name}")

    return line_comment_ranges


def parse_comments_text(file_name,text):
    if text is None:
        return []
    line_comment_ranges = list()
    file_path =  join_path(SOURCE_CODE_OUTPUTS_DIR, file_name)
    source_file_path = write_file(file_path,text,True)
    file_path_srcml = f"{source_file_path}.xml"
    line_comment_ranges = list()

    if any(source_file_path.endswith(e) for e in C_FILE_EXTENSIONS):
        if is_path_exist(file_path_srcml):
            with open(file_path_srcml) as f:
                stdout = f.read()
                stderr = ''
        else:
            stdout, stderr = subprocess_cmd(f'srcml --position {source_file_path}')
            if not stderr:
                write_file(file_path_srcml,stdout)
        if not stderr:
            stdout, stderr = subprocess_cmd(f'srcml --position {source_file_path}')
            for i, line in enumerate(stdout.splitlines()):
                if i == 1 and "<comment " in line.strip() or line.strip().startswith("<comment "):
                    line_comment_ranges.append(CommentRange(start=int(re.search('pos:start="(\d+):', line).groups()[0]),
                                                            end=int(re.search('pos:end="(\d+):', line).groups()[0])))
        else:
            log.warning(f"Error while parsing file {source_file_path}")
    else:
        log.warning(f"File not supported by srcML: {get_file_name_with_parent(source_file_path)}")

    return line_comment_ranges
