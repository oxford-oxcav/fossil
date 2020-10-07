FROM ubuntu:18.04

COPY . ./project

RUN apt-get update &&\
    apt-get install -y python3 python3-pip curl vim &&\
    curl -fsSL 'https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/18.04/install.sh' | bash &&\
    rm -rf ./project/venv &&\
    pip3 install -r ./project/requirements.txt