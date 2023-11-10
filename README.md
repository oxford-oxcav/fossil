# FOSSIL

FOSSIL is a software tool for the sound and automated synthesis of Lyapunov functions and Barrier certificates, for the purposes of verifying the stability or safety of continuous-time dynamical systems modelled as differential equations. The tool leverages a CEGIS-based architecture; the verision 1.0 tool is described in detail by a corresponding [tool paper](https://doi.org/10.1145/3447928.3456646) (also [below](#citation)).

*If you are a repeatability reviewer, please see the [repeatability instructions](repeatability_instructions.md).*

## Requirements

Install:

> python3 python3-pip curl

On Ubuntu 22.04 you can do it with: `sudo apt-get install -y python3 python3-pip curl`

We recommend dReal as an SMT solver. Since we rely on dReal's python interface, dReal's prerequisites must be installed as well. These include Bazel, which must be built from source. dReal provide scripts to install the prerequisites on Ubuntu 18.04, 20.04 and 22.04.

dReal instructions available [on their GitHub repository](https://github.com/dreal/dreal4). The prerequisites for Ubuntu 22.04 can be installed with:

```console
curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/20.04/install.sh | sudo bash
```

Once dReal is installed, you can install the python interface for Fossil 2.0 using:

```console
pip3 install .
```

## Docker

We provide a Docker image of Fossil 2.0 . Begin by installing Docker on your system. Pull the image from Docker Hub:

```console
# docker pull aleccedwards/fossil
# docer run --rm aleccedwards/fossil fossil -h
```

## Citation

To cite Fossil in publications use the following BibTeX entry:

```bibtex
@inproceedings{Abate_Ahmed_Edwards_Giacobbe_Peruffo_2021,
place={Nashville, TN, USA},
series={HSCC ’21},
title={FOSSIL: A Software Tool for the Formal Synthesis of Lyapunov Functions and Barrier Certificates using Neural Networks}, ISBN={978-1-4503-8339-4/21/05},
booktitle={Proceedings of the 24th International Conference on Hybrid Systems: Computation and Control},
publisher={Association for Computing Machinery},
author={Abate, Alessandro and Ahmed, Daniele and Edwards, Alec and Giacobbe, Mirco and Peruffo, Andrea}, year={2021},
month={May},
pages={11},
collection={HSCC ’21} }
```
