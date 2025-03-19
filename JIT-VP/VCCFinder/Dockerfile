FROM python:latest

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libz-dev \
    libconfig-dev \
    libarchive-dev \
    make \
    automake \
    autoconf \
    libtool \
    vim 

RUN git clone https://github.com/rieck/sally

RUN cd sally && ./bootstrap && \
    ./configure && \
    make && \
    make check &&\
    make install

RUN pip install -r requirements.txt