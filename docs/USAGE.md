# How to use

## Input / Ouput format
All input and output files use the JSONL format, where each line is a valid JSON object representing a single instance. See this [**example file**](../sample/sample_train.jsonl).

Some models also require a dictionary file for codes and messages as additional input. By default, it is generated automatically by the data builder feature. However, if you prefer to provide your own dictionary, refer to this example [**dictionary file**](../sample/sample_dict.jsonl).

## Basic usage

### Data builder

```
vulguard mining \
    -dg_save_folder . \
    -mode local \
    -repo_name <PROJECT_NAME> \
    -repo_path <PATH/TO/PROJECT> \
    -repo_clone_url <REPOSITORY_URL> \
    -repo_clone_path <PATH/TO/CLONE> \
    -repo_language <LANGUAGE> \
    -szz vszz \
    -workers 4\
    -vfc_file <PATH/TO/VFC_FILE> #provide your own vfc \
    -lab #specify this to label data in idealized setting \
    -start 0 #First commit to mine, default 0\
    -end 1000 #Last commit to mine, deafault -1\
```

The **data builder process** consists of four main phases:

1. **Mining raw data** – collecting commits and related information from the repository.  
2. **Extracting features** – processing commits to generate code changes, messages and expert features.  
3. **Running the SZZ algorithm** – identifying vulnerability-inducing commits.  
4. **Labeling and splitting data** – preparing data sets.  

This [**example script**](../scripts/test_mining.sh) demonstrates how we construct a commit dataset from the [**libssh2**](https://github.com/libssh2/libssh2) repository.



### Training

```
vulguard training \
    -dg_save_folder . \
    -mode local \
    -repo_name <PROJECT_NAME> \
    -repo_path <PATH/TO/PROJECT> \
    -repo_language <LANGUAGE> \
    -model <MODEL_NAME> \
    -device cuda \
    -threshold 0.5 \
    -model_path <PARENT/DIR/PRETRAINED> \
    -train_set <PATH/TO/TRAIN.JSONL> \
    -val_set <PATH/TO/VAL.JSONL> \
    -hyperparameters <PATH/TO/HYPERPARAMETERS.JSON> \
    -dictionary <PATH/TO/DICT.JSONL> 

```

By default, if not explicitly specified, the **training set (`train_set`)**, **validation set (`val_set`)**, and **dictionaries** are stored in the `dg_cache` directory. These files are automatically generated as outputs of the **data builder** process.  

Each model comes with its own default hyperparameters, defined in the corresponding model folder. For example, the DeepJIT model provides its configuration in: [**vulguard/models/deepjit/hyperparameters.json**](../vulguard/models/deepjit/hyperparameters.json)  

This script [**test_train.sh**](../scripts/test_train.sh) shows how to train a **Logistic Regression** model using the [**sample training data**](../sample/sample_train.jsonl).  


### Evaluating

```
vulguard evaluating\
    -dg_save_folder . \
    -mode local \
    -repo_name <PROJECT_NAME> \
    -repo_path <PATH/TO/PROJECT> \
    -repo_language <LANGUAGE> \
    -model <MODEL_NAME> \
    -device cuda \
    -threshold 0.5 \
    -model_path <PARENT/DIR/PRETRAINED> \
    -test_set <PATH/TO/TEST.JSONL> \
    -size_set <PATH/TO/SIZE.JSONL> \
    -hyperparameters <PATH/TO/HYPERPARAMETERS.JSON> \
    -dictionary <PATH/TO/DICT.JSONL>
```

If not explicitly specified, the **testing set (`test_set`)**, **dictionaries**, and **hyperparameters** will be loaded from their default locations (the same as in the training process).  

In addition, the **`size_set`** parameter points to a [**size file**](../sample/sample_size.jsonl), which records the number of added and deleted lines. This file is used to compute **effort metrics** during evaluation.  

For reference, see the following [**test_evaluate.sh**](../scripts/test_evaluate.sh) script, where we use [**sample testing data**](../sample/sample_test.jsonl) to test the trained **Logistic Regression** model.



### Inferencing

```
vulguard inferencing \
    -dg_save_folder . \
    -mode local \
    -repo_name <PROJECT_NAME> \
    -repo_path <PATH/TO/PROJECT> \
    -repo_language <LANGUAGE> \
    -model <MODEL_NAME> \
    -device cuda \
    -threshold 0.5 \
    -model_path <PARENT/DIR/PRETRAINED> \
    -infer_set <PATH/TO/INFER.JSONL> \
    -size_set <PATH/TO/SIZE.JSONL> \
    -hyperparameters <PATH/TO/HYPERPARAMETERS.JSON> \
    -dictionary <PATH/TO/DICT.JSONL>
```

In the following example [**test_infer.sh**](../scripts/test_infer.sh) script, we infer the trained **Logistic Regression** model with  [**sample inference data**](../sample/sample_infer.jsonl).