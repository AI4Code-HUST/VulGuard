# Installation

## Requirements

VulGuard utilize multiple tools to build data and replicate models'results. Please install these dependencies before using models.

### Core Requirements  
- Python >= 3.7
- Git
- srcML v1.0.0

```
# srcML dependencies
apt-get install -y --reinstall libarchive13 libcurl4 libxml2-dev libxslt1-dev

# Install srcML v1.0.0
wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb && \
    rm -rf libssl1.1_1.1.1f-1ubuntu2_amd64.deb

wget https://github.com/srcML/srcML/releases/download/v1.0.0/srcml_1.0.0-1_ubuntu20.04.deb && \
    dpkg -i srcml_1.0.0-1_ubuntu20.04.deb && \
    rm -rf srcml_1.0.0-1_ubuntu20.04.deb
```

### VCCFinder Requirements
VCCFinder use [sally](https://github.com/rieck/sally) to vectorize extracted features for their Support Vector Machine models. 

```
# sally dependencies
apt-get install -y libz-dev libconfig-dev libarchive-dev make automake autoconf libtool

# Install sally
git clone https://github.com/rieck/sally

cd sally && ./bootstrap && \
    ./configure && \
    make && \
    make check &&\
    make install
```

### Graph Builder Requirements
CodeJIT use [Joern](https://github.com/joernio/joern) build code graph for their Graph Neural Network models. 

```
# joern dependencies jdk11, gcc
apt-get install -y gcc

mkdir -p dependencies/jdk && \
    wget https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz && \
    tar -xvzf "openjdk-11.0.2_linux-x64_bin.tar.gz" -C dependencies/jdk && \
    rm "openjdk-11.0.2_linux-x64_bin.tar.gz"

export PATH="dependencies/jdk/jdk-11.0.2/bin:${PATH}"


# Install Joern
mkdir -p dependencies/joern && \
    wget https://github.com/joernio/joern/releases/download/v1.1.1298/joern-cli.zip && \
    unzip joern-cli.zip -d dependencies/joern && \
    chmod -R u+x dependencies/joern/joern-cli &&\
    rm joern-cli.zip

export PATH="dependencies/joern/joern-cli:${PATH}"
```

## Installation

We offer installation via Docker and via Python packages. We encourage installation via Docker as all tools and dependencies would be installed. If you want to install from scratch using Python packages, please install your desired requirements.

### Install via Docker

**GPU-supported** 

```
# Install nvidia-container-toolkit
bash scripts/setup_container_toolkit.sh

# Build and exec vulguard container
docker compose -f docker-compose.gpu.yml up -d --build
docker exec -it vulguard-gpu /bin/bash
```

**CPU-only**
 ```
docker-compose -f docker-compose.cpu.yml up -d --build
docker exec -it vulguard-cpu /bin/bash
```


### Install via Python packages

**Create and activate a virtual environment**

```
# create a virtual environment
python3.11 -m venv env

# activate the environment
. env/bin/activate
```

**Install VulGuard from PyPI or Build VulGuard from source**

```bash
# install BARO from PyPI
pip install vulguard

# build BARO from source
pip install -e .
```