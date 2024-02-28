# FOSSIL

FOSSIL is a software tool for the sound and automated synthesis of Lyapunov functions and Barrier certificates, for the purposes of verifying the stability or safety of continuous-time dynamical systems modelled as differential equations. The tool leverages a CEGIS-based architecture; the verision 1.0 tool is described in detail by a corresponding [tool paper](https://doi.org/10.1145/3447928.3456646) (also [below](#citation)).

## Requirements

Fossil 2.0 requires Python 3.9 or later.
We recommend dReal as an SMT solver. Since we rely on dReal's python interface, dReal's prerequisites must be installed as well. These include Bazel, which must be built from source. dReal provide scripts to install the prerequisites on Ubuntu 18.04, 20.04 and 22.04.

dReal instructions available [on their GitHub repository](https://github.com/dreal/dreal4). The prerequisites for Ubuntu 22.04 can be installed with:

```console
curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install_prereqs.sh | sudo bash
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

## Command-line Interface

Fossil 2.0 provides a command-line interface, which relies on yaml files to specify the configuration of the synthesis problem. The following configuration
file specifies a stability verification problem for a continuous-time model with two variables, and a neural network with two hidden layers of 5 neurons each.
It also specifies the synthesis of a controller with one hidden layer of 5 neurons to stabilize the system. Further details on the options available in the configuration file can be found in the [user guide](user-guide.md), and examples found in the [experiments folder](experiments/benchmarks/cli).

```yaml
N_VARS: 2
SYSTEM: [x1 - x0**3, u0]
CERTIFICATE: Lyapunov
TIME_DOMAIN: CONTINUOUS
DOMAINS:
  XD: Torus([0,0], 1.0, 0.01)
N_DATA:
  XD: 1000
N_HIDDEN_NEURONS: [5, 5]
ACTIVATION: [SIGMOID, SQUARE]
CTRLAYER: [5,1]
CTRLACTIVATION: [LINEAR]
VERIFIER: DREAL
```

The following command will synthesize a Lyapunov function and a controller for the system specified in the configuration file:

```console
fossil config.yaml
```

## Citations

Fossil has been described in the following papers:

### Version 2.0

#### Fossil 2.0: Formal Certificate Synthesis for the Verification and Control of Dynamical Models

* Counterexampl-Guided synthesis of neural network certificates and controllers
* Range of properties including stability, safety, reach-while-avoid for continuous-time models
* Stability and safety for discrete-time models
* Pythonic and command-line interfaces

```bibtex
@inproceedings{Edwards_Peruffo_Abate_2024,
series={HSCC ’24},
title={Fossil 2.0: Formal Certificate Synthesis for the Verification and Control of Dynamical Models}, 
booktitle={Proceedings of the 27th International Conference on Hybrid Systems: Computation and Control},
publisher={Association for Computing Machinery},
author={Edwards, Alec and Peruffo, Andrea and Abate, Alessandro}, year={2021},
collection={HSCC ’24} }
```

### Version 1.0

#### FOSSIL: A Software Tool for the Formal Synthesis of Lyapunov Functions and Barrier Certificates using Neural Networks

* Counterexample-Guided Inductive Synthesis (CEGIS) for Lyapunov functions and Barrier certificates
* Continuous-time models
* Leverages neural networks as candidate Lyapunov functions and Barrier certificates

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
