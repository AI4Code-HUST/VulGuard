# VulGuard

This is the repository of VulGuard, An Unified Tool for Evaluating Just-In-Time Vulnerability Prediction Models.

## Installation

## via Docker (RECOMMENDED)
```
docker compose up --build -d
docker exec -it vulguard /bin/bash
```

Inside docker container:

```
python setup.py develop
```

### If you want Docker container to access GPU(s), please download `nvidia-container-toolkit`

**Note**: download this outside of the container

Install the `nvidia-container-toolkit` package as per official documentation at Github.

We also provide [a quick-run script](scripts/setup_nvidia_container_toolkit.sh) for Debian-based OS

### From scratch

- SrcML
```
# Install libarchive13 libcurl4 libxml2
sudo apt-get install libarchive13 libcurl4 libxml2

# Install libssl
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb && \
    rm -rf libssl1.1_1.1.1f-1ubuntu2_amd64.deb

# Install SrcML
RUN wget http://131.123.42.38/lmcrs/v1.0.0/srcml_1.0.0-1_ubuntu20.04.deb && \
    dpkg -i srcml_1.0.0-1_ubuntu20.04.deb && \
    rm -rf srcml_1.0.0-1_ubuntu20.04.deb
```

- Dependencies
```
pip install -r requirements.txt
```

- Setup VulGuard
```
python setup.py develop
```

## Usages

### Mining commits from Git repositories

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
