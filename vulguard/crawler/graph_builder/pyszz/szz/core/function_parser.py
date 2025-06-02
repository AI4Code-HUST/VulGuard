import logging as log
import re
from collections import namedtuple

from config import C_FILE_EXTENSIONS, SOURCE_CODE_OUTPUTS_DIR
from file_manager import get_file_name_with_parent, write_file, join_path, is_path_exist
from helpers import subprocess_cmd

CommentRange = namedtuple('CommentRange', 'start end')
# srcml_file_ext = ['.c', '.h', '.hh', '.hpp', '.hxx', '.cxx', '.cpp', '.cc', '.cs', '.java']

__CACHE_FUNCTIONS = {}


def parse_functions(source_file_path: str):
    if source_file_path not in __CACHE_FUNCTIONS:
        list_functions = parse_functions_srcml(source_file_path)
        __CACHE_FUNCTIONS[source_file_path] = list_functions
    list_functions = __CACHE_FUNCTIONS[source_file_path]
    return list_functions


def parse_functions_srcml(source_file_path):
    list_functions = list()

    if any(source_file_path.endswith(e) for e in C_FILE_EXTENSIONS):
        source_file_path_srcml_parsed = f"{source_file_path}.xml"
        
        if is_path_exist(source_file_path_srcml_parsed):
            with open(source_file_path_srcml_parsed) as f:
                    stdout = f.read()
                    stderr = ''
        else:
            stdout, stderr = subprocess_cmd(
                f'srcml --position {source_file_path}')
        
        if not stderr:
            stdout = stdout.replace('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '')
            stdout = stdout[stdout.index(">")+1:]
            outlines =  stdout.splitlines()
            for i, line in enumerate(outlines):
                if len(line) > 0 and len(line[0].strip()) != 0 and line.find("<") > 0:
                    line = line[line.index("<"):]
                if ">class<" in line:
                    continue
                if "<function " in line or "<constructor " in line or "<destructor" in line or\
                    ((line.startswith("<decl_stmt ") or (("<decl_stmt " in line or "<macro " in line) and "<block" in line)) and
                        ">;<" not in line and "<argument_list" in line) or \
                        ((line.startswith("<macro ") or line.startswith("<expr_stmt ")) and ">;<" not in line):
                    if "</macro> <block" in line:
                        nLine = line[line.index("</macro> <block"):]
                        start = int(re.search('pos:start="(\d+):', line).groups()[0])
                        end = int(re.search('pos:end="(\d+):', nLine).groups()[0])
                    elif line.startswith("<macro ") and line.endswith("</macro>") and \
                            i+1 < len(outlines) and outlines[i+1].startswith("<block "):
                        nLine = outlines[i+1]
                        start = int(re.search('pos:start="(\d+):', line).groups()[0])
                        end = int(re.search('pos:end="(\d+):', nLine).groups()[0])
                    elif line.startswith("<macro ") and "</macro>" not in line:
                        c = 1
                        while "</macro>" not in outlines[i+c] or "<block" not in outlines[i+c]:
                            c += 1
                            if i + c >= len(outlines)-1:
                                break
                        if c > 5:
                            start = 0
                            end = 0
                        else:
                            if "<block " in outlines[i+c]:
                                nLine = outlines[i+c][outlines[i+c].index("<block "):]
                            else:
                                nLine = line
                            start = int(re.search('pos:start="(\d+):', line).groups()[0])
                            end = int(re.search('pos:end="(\d+):', nLine).groups()[0])
                    else:
                        start = int(re.search('pos:start="(\d+):', line).groups()[0])
                        if "<block" in line:
                            nLine = line[line.index("<block "):]
                            end = int(re.search('pos:end="(\d+):', nLine).groups()[0])
                        else:
                            end = int(re.search('pos:end="(\d+):', line).groups()[0])
                    if start >= end:
                        continue
                    pad = 5
                    if end - start <= 5:
                        pad = end - start
                    function_text = "".join(stdout.splitlines()[start-1:start+pad])
                    if "</template>" in function_text:
                        function_text = function_text[function_text.find(
                            "</template>")+15:]
                    elif ">template<" in function_text:
                        function_text = function_text[function_text.find(
                            ">template<")+20:]
                    function_text = function_text[:function_text.find("<block")]
                    
                    token_end = "<argument_list"
                    if "<function" in line or "<constructor" in line or "<destructor" in line:
                        token_end = "<parameter_list"
                    elif line.startswith("<expr_stmt ") and len(outlines[i]) > 0 and\
                            outlines[i][0] != " " and outlines[i][0] != "<":
                        token_end = "<expr_stmt"
                        function_text = ">" + function_text
                    if token_end not in function_text:
                        continue
                    function_text = function_text[:function_text.index(token_end)+20]
                    token_start = ""
                    try:
                        name_xml = re.search(f"{token_start}(.)*{token_end}",function_text)
                        re_name = re.findall(">([A-z0-9_:]+)<",name_xml.group())
                        name = ''.join(re_name)
                    except:
                        print("="*33)
                        print(line)
                        print(source_file_path_srcml_parsed)
                        continue
                    if len(list_functions) > 0 and list_functions[-1]["end_line"] > start:
                        # print("Error parser function: " + list_functions[-1]["name"])
                        list_functions[-1]["end_line"] = start -1
                    list_functions.append(
                        {"name": name, "start_line": start, "end_line": end})
            write_file(source_file_path_srcml_parsed, stdout, True)
        else:
            log.warning(f"Error while parsing file {source_file_path}")
    else:
        log.warning(
            f"File not supported by srcML: {get_file_name_with_parent(source_file_path)}")

    return list_functions


def parse_functions_content(file_name, text):
    if text is None:
        return []
    text = text.replace('\\\n',"\n//")
    text = text.replace('\n#',"\n//")
    text = text.replace("~","")
    text = text.replace("static ","")#explicit 
    text = text.replace("explicit ","")#explicit 
    text = text.replace(" * "," ")
    text = re.sub('\n([ ])+\"(.)+\"\);',"\n//"+"X",text)
    
    regex = '\n(.*){[A-z0-9_:.,\" ]*}(.*)\n'
    res = re.search(regex,text)
    while res is not None:
        text = text.replace(res.group(),"\n" + " ".join(res.groups()) + "\n")
        res = re.search(regex,text)
    
    file_path = join_path(SOURCE_CODE_OUTPUTS_DIR, file_name)
    file_path = write_file(file_path, text, True)
    return parse_functions(file_path)
