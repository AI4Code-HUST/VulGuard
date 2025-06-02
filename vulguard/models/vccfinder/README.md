# VCC Finder

This is the replication of VCC Finder.

## Requirements
- [SALLY](https://github.com/rieck/sally)
- Python 3

## Installation

### via Docker (RECOMMENDED)
```
docker compose up --build -d
docker exec -it sally /bin/bash
```

### From scratch

- Sally dependecies:
```
RUN apt-get update && apt-get install -y \
    gcc \
    libz-dev \
    libconfig-dev \
    libarchive-dev \
    make \
    automake \
    autoconf \
    libtool
```

- Sally:
```
git clone https://github.com/rieck/sally && \
    cd sally && \
    ./bootstrap && \
    ./configure && \
    make && \
    make check &&\
    make install
```

- Dependencies:
```
pip install -r requirements.txt
```

### Usages:
- Vectorized Data using Sally:
```
# Chage the path to data files in preprocess.py
python preprocess.py
```

- Train and Evaluate:
```
# Chage the path to data files in run.py
python run.py
```