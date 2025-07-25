# How to use

## Input / Ouput format
ALl input and output files are in jsonl format, in which each line is a json string, represent an instance.

## Basic usage

### Data builder

```
vulguard mining \
    -repo_name <project_name> \
    -repo_path <path/to/project> \
    -mode <local or remote> \
    -repo_language <main_language_of_project> \
    -szz <szz_algorithm_name> \
    -workers <number_of_parallel_miners>
```

[Example](scripts/test_mining.sh)

### Training

```
vulguard training  \
    -model <model_name> \
    -train_set <path/to//train/set> \
    -val_set <path/to/val/set> \
    -dictionary <path/to/dictionary> \
    -dg_save_folder <path/to/save/folder> \
    -repo_name <project_name> \
    -repo_language <main_language_of_project> \
    -device cuda \
    -epoch <number_of_epochs>
```

[Example](scripts/test_train.sh)

### Evaluating

```
vulguard evaluating  \
    -model <model_name> \
    -test_set <path/to/test/set> \
    -dictionary <path/to/dictionary> \
    -dg_save_folder <path/to/save/folder> \
    -repo_name <project_name> \
    -repo_language <main_language_of_project> \
    -device cuda 
```

[Example](scripts/test_evaluate.sh)


### Inferencing

```
vulguard inferencing  \
    -model <model_name> \
    -model_path <path/to/trained/model> \
    -infer_set <path/to/infer/set> \
    -dictionary <path/to/dictionary> \
    -dg_save_folder <path/to/save/folder> \
    -device cuda 
```

[Example](scripts/test_infer.sh)