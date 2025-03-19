import pandas as pd
from pydriller import Git as PyDrillerGitRepo, ModifiedFile, Commit

from config import RAW_COMMIT_DATA_DIR, FIXING_COMMIT_FILE_PATH, C_FILE_EXTENSIONS, REPOSITORY_COMMITS_DIR, \
    REPOSITORY_DATA_FOLDER_NAME, DIFF_OUTPUTS_FOLDER_NAME, JOERN_PARSED_OUTPUTS_FOLDER_NAME, \
    DEPENDENCY_OUTPUTS_FOLDER_NAME, DEPENDENCY__OUTPUTS_DIR, MAPPING_FILE_EXT, DEPENDENCY_FILE_EXT
from file_manager import get_file_name, get_cloned_repository, write_file, join_path, get_outer_dir, is_path_exist, \
    find_all_files_by_wildcard, lock_dir
from helpers import get_logger
from joern.diff import write_unchanged_line_mapping_after_diff, find_modified_lines_in_commit_changes, \
    read_mapping_line_file_path
from joern.joern_dependency_extractor import joern_graph_extraction
from joern.joern_parser import run_joern
from szz.core.abstract_szz import ImpactedFile
from szz.core.comment_parser import parse_comments

logger = get_logger(__name__)


def main():
    # df = pd.read_csv(FIXING_COMMIT_FILE_PATH)
    
    # grouped_commits_by_repo = df["commit_id"].apply(list)
    # for project_name, commit_ids in grouped_commits_by_repo.iteritems():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--file", type=str)
    parser.add_argument("--project", type=str)
    p = parser.parse_args()
    
    if p.file == "vic":
        FILE = join_path(RAW_COMMIT_DATA_DIR, "VIC.jsonl")
    elif p.file == "vfc":
        FILE = join_path(RAW_COMMIT_DATA_DIR, "VFC.jsonl")
    else:
        FILE = join_path(RAW_COMMIT_DATA_DIR, "non_VIC.jsonl")
        
    df = pd.read_json(FILE, lines=True)
    commit_ids = df["commit_id"].to_list()[p.begin:p.end]   
    repo_dir = get_cloned_repository(p.project)

    # try:
    #     lock_dir(repo_dir)
    # except BlockingIOError as e:
    #     continue
    run_finding_impacted_dependence_lines(repo_dir, commit_ids)
    # break

def run_finding_impacted_dependence_lines(repo_dir, commit_ids):
    repo_name = get_file_name(repo_dir)
    # repo_obj = Repo(repo_dir)

    logger.info(f"Saving changed C/C++ files from [{len(commit_ids)}] fixing commits in [{repo_name}]")
    repo_obj = PyDrillerGitRepo(repo_dir)

    for commit_id in commit_ids:
        # if commit_id != "2e1198672759eda6e122ff38fcf6df06f27e0fe2":
        #     continue
        commit_obj = repo_obj.get_commit(commit_id)

        for mod in commit_obj.modified_files:
            # skip files are added, deleted or only renamed
            if mod.source_code is None or mod.source_code_before is None:
                continue

            # filter files by extension
            ext = mod.filename.rsplit('.', 1)
            if len(ext) < 2 or (len(ext) > 1 and ext[1] not in C_FILE_EXTENSIONS):
                # logger.warning(f"skip file: {mod.filename}")
                continue

            new_file_path, old_file_path, mapping_file_path = write_diff_line_mapping(repo_name=repo_name,
                                                                                      commit_obj=commit_obj,
                                                                                      mod_file=mod)
            write_joern_parsed_output(repo_name=repo_name, commit_obj=commit_obj, mod_file=mod)
            write_impacted_dependence_lines(repo_name=repo_name, commit_obj=commit_obj, mod_file=mod,
                                            new_file_path=new_file_path, diff_mapping_file_path=mapping_file_path)
            # break
        # break


def write_joern_parsed_output(repo_name: str, commit_obj: Commit, mod_file: ModifiedFile):
    commit_id = commit_obj.hash
    new_file_path = join_path(REPOSITORY_COMMITS_DIR, repo_name, commit_id, mod_file.new_path)
    output_joern_dir = get_outer_dir(
        new_file_path.replace(REPOSITORY_DATA_FOLDER_NAME, JOERN_PARSED_OUTPUTS_FOLDER_NAME))
    run_joern(new_file_path, output_joern_dir)


def write_diff_line_mapping(repo_name: str, commit_obj: Commit, mod_file: ModifiedFile):
    commit_id = commit_obj.hash
    parent_commit_id = str(commit_obj.parents[0])

    new_file_path = write_file(join_path(REPOSITORY_COMMITS_DIR, repo_name, commit_id, mod_file.new_path),
                               mod_file.source_code,
                               skip_if_existed=True)
    old_file_path = write_file(
        join_path(REPOSITORY_COMMITS_DIR, repo_name, parent_commit_id, mod_file.old_path),
        mod_file.source_code_before, skip_if_existed=True)

    mapping_output_file_path = new_file_path.replace(REPOSITORY_DATA_FOLDER_NAME,
                                                     DIFF_OUTPUTS_FOLDER_NAME) + MAPPING_FILE_EXT
    if not is_path_exist(mapping_output_file_path):
        write_unchanged_line_mapping_after_diff(new_file_path, old_file_path, mapping_output_file_path)

    return new_file_path, old_file_path, mapping_output_file_path


def write_impacted_dependence_lines(repo_name: str, commit_obj: Commit, new_file_path: str, mod_file: ModifiedFile,
                                    diff_mapping_file_path: str):
    """Get lines that are dependent on added lines. """
    output_dep_file_path = diff_mapping_file_path.replace(DIFF_OUTPUTS_FOLDER_NAME,
                                                          DEPENDENCY_OUTPUTS_FOLDER_NAME).replace(MAPPING_FILE_EXT,
                                                                                                  DEPENDENCY_FILE_EXT)
    if is_path_exist(output_dep_file_path):
        return
    commit_id = commit_obj.hash
    logger.info(f"Writing impacted lines from [{commit_id}] [{get_file_name(new_file_path)}]")
    modified_line_nums = find_modified_lines_in_commit_changes(mod_file, diff_mapping_file_path)
    raw_added_line_nums = set([added[0] for added in mod_file.diff_parsed["added"]])
    actual_added_line_nums = list(raw_added_line_nums - modified_line_nums)
    filtered_comment_added_line_nums = []
    code_comment_ranges = parse_comments(new_file_path)
    for added_ln in actual_added_line_nums:
        for comment_range in code_comment_ranges:
            if comment_range.start <= added_ln <= comment_range.end:
                continue
            filtered_comment_added_line_nums.append(added_ln)
    actual_added_line_nums = filtered_comment_added_line_nums

    try:
        graph_extraction_results = joern_graph_extraction(source_file_path=new_file_path,
                                                          joern_file_path=new_file_path.replace(
                                                              REPOSITORY_DATA_FOLDER_NAME,
                                                              JOERN_PARSED_OUTPUTS_FOLDER_NAME))
    except Exception as e:
        logger.warning(f"[{repo_name}] Failed to get extracted joern graph [{commit_id}]"
                       f"[{get_file_name(new_file_path)}]")
        return

    after_graph = graph_extraction_results[0]

    # Get nodes in graph corresponding to real added lines
    added_after_lines = after_graph[after_graph.id.isin(actual_added_line_nums)]

    # Get lines dependent on added lines in added graph
    dep_line_num_dict = {}
    for row in added_after_lines.iterrows():
        r = row[1]
        try:
            dep_line_num_dict[r.id] = set(r.data + r.control)
        except Exception as e:
            logger.warning(f"[{repo_name}] Failed to find dependencies of line [{commit_id}]"
                           f"[{get_file_name(new_file_path)}#L{r.id}]")
            dep_line_num_dict[r.id] = set()

    mapping_line_dict = read_mapping_line_file_path(diff_mapping_file_path)
    output_dep_list = []

    for added_ln, dep_lns in dep_line_num_dict.items():
        for dep_ln in dep_lns:
            dep_ln = int(dep_ln)
            if dep_ln in mapping_line_dict:
                output_dep_list.append((added_ln, dep_ln, mapping_line_dict[dep_ln]))

    output_dep_file_content = f"{mod_file.old_path}\n"
    output_dep_file_content += "added_line->line_dep_on_added_line_in_new_file->line_dep_on_added_line_in_old_file\n"
    for item in output_dep_list:
        output_dep_file_content += f"{item[0]}->{item[1]}->{item[2]}\n"
    write_file(output_dep_file_path, output_dep_file_content, skip_if_existed=False)


def get_impacted_files_by_dependencies(repo_name: str, fixing_commit_hash: str):
    dependency_dir_by_commit = join_path(DEPENDENCY__OUTPUTS_DIR, repo_name, fixing_commit_hash)
    if not is_path_exist(dependency_dir_by_commit):
        return None
    all_dep_file_paths = find_all_files_by_wildcard(dependency_dir_by_commit, f"**/*{DEPENDENCY_FILE_EXT}",
                                                    recursive=True)
    impacted_files = []
    for f_path in all_dep_file_paths:
        ext = f_path.rsplit('.', 2)
        if len(ext) < 3 or (len(ext) > 1 and ext[1] not in C_FILE_EXTENSIONS):
            continue
        f_lines = open(f_path).readlines()
        old_file_path = f_lines[0].strip()
        dep_lines = [int(l.strip().split("->")[2]) for l in f_lines[2:]]
        if len(dep_lines) > 0:
            impacted_files.append(ImpactedFile(old_file_path, dep_lines))
    return impacted_files


if __name__ == "__main__":
    main()
