# JIT-DP models

This is the replication of JIT-DP models.

## Installation

## via Docker (RECOMMENDED)
```
docker compose up --build -d
docker exec -it defectguard /bin/bash
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

- Setup DefectGuard
```
python setup.py develop
```

## Usages

### Mining commits from Git repositories

```
defectguard mining \
    -repo_name <project_name> \
    -repo_path <path/to/project> \
    -repo_language <main_language_of_project> \
    -pyszz_path <path/to/project/pyszz_v2> \
    -szz <szz_algorithm_name> \
    -workers <number_of_parallel_miners>
```

[Example](scripts/test_mining.sh)

### Training

```
defectguard training  \
    -model <model_name> \
    -feature_train_set <path/to/expert/feature/train/set> \
    -commit_train_set <path/to/expert/commit/train/set> \
    -commit_val_set <path/to/expert/feature/train/set> \
    -dictionary <path/to/dictionary> \
    -dg_save_folder <path/to/save/folder> \
    -repo_name <project_name> \
    -device cuda \
    -repo_language <main_language_of_project> \
    -epoch <number_of_epochs>
```

[Example](scripts/test_train.sh)

### Evaluating

```
defectguard evaluating  \
    -model <model_name> \
    -feature_test_set <path/to/expert/feature/train/set> \
    -commit_test_set <path/to/expert/commit/train/set> \
    -dictionary <path/to/dictionary> \
    -dg_save_folder <path/to/save/folder> \
    -repo_name <project_name> \
    -device cuda \
    -repo_language <main_language_of_project> \
```

[Example](scripts/test_evaluate.sh)
