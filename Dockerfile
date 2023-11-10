# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Update and install required packages in a single step
RUN apt-get update \
    && apt-get install -y python3 python3-pip curl nano \
    && curl -fsSL 'https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install_prereqs.sh' | bash \
    && apt-get remove -y bazel bison flex g++ wget \
    && apt-get autoclean -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache/bazel

# Installing python packages
RUN pip3 install torch==2.0.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copying the application source should be one of the last steps
WORKDIR /fossil
COPY . .
RUN pip3 install .

