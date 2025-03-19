# CodeJIT Graph Builder 

This is the implemtation to build CTG for CodeJIT for any commit without labeling.

## Requirements:
- Joern v1.1.1193
- srcML
- git
- jdk 11
- Python 3.7

## Input preparation:
- INPUT FILE (jsonl files contain ids of commmits you want to build graph):  place in `./data/vul_commit_database`
- GIT REPOSITORIES (including `.git`): clone to `./data/git_repository_data/cloned_repositories`

## Usages
- Run this script
```
bash run.sh <project_name> <input_file_name> <first_id> <last_id>
```

Example:
```
bash run.sh FFmpeg test.jsonl 0 2
```