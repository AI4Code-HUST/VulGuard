FROM ubuntu:22.04

WORKDIR /app

COPY . /app

RUN apt-get update

# Install srcML
RUN apt-get install -y --reinstall wget libarchive13 libcurl4 libxml2

RUN wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb && \
    rm -rf libssl1.1_1.1.1f-1ubuntu2_amd64.deb

RUN wget http://131.123.42.38/lmcrs/v1.0.0/srcml_1.0.0-1_ubuntu20.04.deb && \
    dpkg -i srcml_1.0.0-1_ubuntu20.04.deb && \
    rm -rf srcml_1.0.0-1_ubuntu20.04.deb

ENV TZ=US/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y python3 python3-pip git libxml2-dev libxslt1-dev
RUN pip install -r cpu-only-requirements.txt