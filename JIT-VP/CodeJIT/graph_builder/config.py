import pathlib
import sys

from file_manager import join_path, mkdir_if_not_exist

BASE_DIR = str(pathlib.Path(__file__).parent.absolute()) + "/"

BASE_DATA_DIR = join_path(BASE_DIR, "data")

DEBUG = True

# ------ START CONFIG INPUT ------
RAW_COMMIT_DATA_DIR = join_path(BASE_DATA_DIR, "vul_commit_database")
mkdir_if_not_exist(RAW_COMMIT_DATA_DIR)

if DEBUG:
    FIXING_COMMIT_FILE_NAME = "test.jsonl"
else:
    FIXING_COMMIT_FILE_NAME = "filtered_c_lang__fixing_commits.csv"

FIXING_COMMIT_FILE_PATH = join_path(RAW_COMMIT_DATA_DIR, FIXING_COMMIT_FILE_NAME)
# ------ END CONFIG INPIT ------

# ------ START CONFIG OUTPUT ------
JIT_VUL_DATA_DIR = join_path(BASE_DATA_DIR, "jit_vul_dataset")
mkdir_if_not_exist(JIT_VUL_DATA_DIR)

JIT_VUL_CONTRIBUTING_DATA_FILE_NAME = "vul_contributing_commit_data.jsonl"
JIT_VUL_TRIGGERING_DATA_FILE_NAME = "vul_triggering_commit_data.csv"
JIT_VUL_CLEAN_DATA_FILE_NAME = "vul_clean_commit_data.csv"
JIT_VUL_CONTRIBUTING_COMMIT_FILE_NAME = "vul_contributing_commit_data.csv"
JIT_VUL_UNLABELLED_DATA_FILE_NAME = "vul_unlabelled_commit_data.csv"
JIT_VUL_META_DATA_FILE_NAME = "vul_metadata_commit_data.csv"

JIT_VUL_CONTRIBUTING_DATA_FILE_PATH = join_path(JIT_VUL_DATA_DIR, JIT_VUL_CONTRIBUTING_DATA_FILE_NAME)
JIT_VUL_TRIGGERING_DATA_FILE_PATH = join_path(JIT_VUL_DATA_DIR, JIT_VUL_TRIGGERING_DATA_FILE_NAME)
JIT_VUL_CLEAN_DATA_FILE_PATH = join_path(JIT_VUL_DATA_DIR, JIT_VUL_CLEAN_DATA_FILE_NAME)
JIT_VUL_UNLABELLED_DATA_FILE_PATH = join_path(JIT_VUL_DATA_DIR, JIT_VUL_UNLABELLED_DATA_FILE_NAME)
JIT_VUL_CONTRIBUTING_COMMIT_FILE_PATH = join_path(JIT_VUL_DATA_DIR, JIT_VUL_CONTRIBUTING_COMMIT_FILE_NAME)
JIT_VUL_METADATA_COMMIT_FILE_PATH = join_path(JIT_VUL_DATA_DIR, JIT_VUL_META_DATA_FILE_NAME)


# ------ END CONFIG OUTPUT ------


# ------ START CONFIG OTHER ------
REPOSITORY_DATA_FOLDER_NAME = "git_repository_data"
BASE_REPOSITORY_DATA_DIR = join_path(BASE_DATA_DIR, REPOSITORY_DATA_FOLDER_NAME)
CLONED_REPOSITORIES_DIR = join_path(BASE_REPOSITORY_DATA_DIR, "cloned_repositories")
mkdir_if_not_exist(CLONED_REPOSITORIES_DIR)

REPOSITORY_COMMITS_DIR = join_path(BASE_REPOSITORY_DATA_DIR, "commits")
mkdir_if_not_exist(REPOSITORY_COMMITS_DIR)

JOERN_PARSED_OUTPUTS_FOLDER_NAME = "joern_parsed_output"
JOERN_PARSED_OUTPUTS_DIR = join_path(BASE_DATA_DIR, JOERN_PARSED_OUTPUTS_FOLDER_NAME, "commits")
mkdir_if_not_exist(REPOSITORY_COMMITS_DIR)
JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR = join_path(BASE_DATA_DIR, JOERN_PARSED_OUTPUTS_FOLDER_NAME, "functions")
mkdir_if_not_exist(JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR)


DIFF_OUTPUTS_FOLDER_NAME = "diff_output"
DIFF_OUTPUTS_DIR = join_path(BASE_DATA_DIR, DIFF_OUTPUTS_FOLDER_NAME, "commits")
MAPPING_FILE_EXT = ".mapping"
mkdir_if_not_exist(DIFF_OUTPUTS_DIR)

DEPENDENCY_OUTPUTS_FOLDER_NAME = "dependency_output2"
DEPENDENCY__OUTPUTS_DIR = join_path(BASE_DATA_DIR, DEPENDENCY_OUTPUTS_FOLDER_NAME, "commits")
DEPENDENCY_FILE_EXT = ".dep"
mkdir_if_not_exist(DEPENDENCY__OUTPUTS_DIR)

COMMIT_EXTRACT_OUTPUTS_FOLDER_NAME = "extracted_output"
COMMIT_EXTRACT_OUTPUTS_DIR = join_path(BASE_DATA_DIR, COMMIT_EXTRACT_OUTPUTS_FOLDER_NAME, "commits")
METADATA_EXTRACT_OUTPUTS_DIR = join_path(BASE_DATA_DIR, COMMIT_EXTRACT_OUTPUTS_FOLDER_NAME, "metadata")
PROJECT_EXTRACT_OUTPUTS_DIR = join_path(BASE_DATA_DIR, COMMIT_EXTRACT_OUTPUTS_FOLDER_NAME, "project")
SOURCE_CODE_OUTPUTS_DIR = join_path(BASE_DATA_DIR, COMMIT_EXTRACT_OUTPUTS_FOLDER_NAME, "source_code")
mkdir_if_not_exist(COMMIT_EXTRACT_OUTPUTS_DIR)
mkdir_if_not_exist(PROJECT_EXTRACT_OUTPUTS_DIR)
mkdir_if_not_exist(METADATA_EXTRACT_OUTPUTS_DIR)
mkdir_if_not_exist(SOURCE_CODE_OUTPUTS_DIR)
# ------ END CONFIG OTHER ------

C_FILE_EXTENSIONS = {"c", "C", "c++", "C++", "cc", "CC", "cp", "CP", "cpp", "CPP", "cppm", "CPPM", "cxx", "CXX", "h",
                     "H", "hdl", "HDL", "hpp", "HPP", "hxx", "HXX", "ixx", "IXX"}

sys.path.append(join_path(BASE_DIR, 'pyszz'))


# -------- ORTHER ------------

TOKEN_SPLIT_FUNCTION = "___SPLIT_FUNCTION___"

MAX_CHANGE = 100