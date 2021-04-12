# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

FROM ubuntu:18.04

COPY . ./project

RUN apt-get update &&\
    apt-get install -y python3 python3-pip curl vim &&\
    curl -fsSL 'https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/18.04/install.sh' | bash &&\
    rm -rf ./project/venv &&\
    pip3 install -r ./project/requirements.txt
