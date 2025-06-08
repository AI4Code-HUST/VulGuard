FROM ubuntu:22.04

WORKDIR /app

COPY . /app

RUN apt-get update
RUN apt-get install -y python3 python3-pip git vim wget zip unzip

# Install srcML
RUN apt-get install -y --reinstall libarchive13 libcurl4 libxml2-dev libxslt1-dev

RUN wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb && \
    rm -rf libssl1.1_1.1.1f-1ubuntu2_amd64.deb

RUN wget http://131.123.42.38/lmcrs/v1.0.0/srcml_1.0.0-1_ubuntu20.04.deb && \
    dpkg -i srcml_1.0.0-1_ubuntu20.04.deb && \
    rm -rf srcml_1.0.0-1_ubuntu20.04.deb

# Install sally
RUN apt-get install -y gcc libz-dev libconfig-dev libarchive-dev make automake autoconf libtool

RUN git clone https://github.com/rieck/sally

RUN cd sally && ./bootstrap && \
    ./configure && \
    make && \
    make check &&\
    make install

# Install jdk 11
RUN mkdir /opt/java && \
    wget https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz && \
    tar -xvzf "openjdk-11.0.2_linux-x64_bin.tar.gz" -C /opt/java 
ENV PATH="/opt/java/jdk-11.0.2/bin:${PATH}"

# Install Joern
RUN wget https://github.com/joernio/joern/releases/download/v1.1.1298/joern-cli.zip && \
    unzip joern-cli.zip -d /opt/joern && \
    chmod -R u+x /opt/joern/joern-cli &&\
    rm joern-cli.zip
ENV PATH="/opt/joern/joern-cli:${PATH}"

ENV TZ=US/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN pip install --ignore-installed -r requirements.txt
RUN python3 setup.py develop