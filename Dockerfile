# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

FROM ubuntu:20.04

COPY . ./project

RUN apt-get update &&\
    apt install software-properties-common -y &&\
    add-apt-repository ppa:deadsnakes/ppa &&\
    apt-get install -y python3.8 python3-pip python3.8-dev curl vim libjpeg-dev &&\
    curl -fsSL 'https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/20.04/install.sh' | bash &&\
    rm -rf ./project/venv &&\
    pip3 install -r ./project/requirements.txt
