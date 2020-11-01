# Requirements

Install:

> python3 python3-pip curl

On Ubuntu you can do it with: `apt-get install -y python3 python3-pip curl`

Run: `pip3 install -r ./requirements.txt`

dReal instructions available [on their GitHub repository](https://github.com/dreal/dreal4).


## Jupyter Notebooks

To run the Jupyter Notebook playgrounds, use:

> jupyter notebook experiments/Fossil-playground.ipynb

Users who wish to use the Jupyter Notebooks with a python virtual environment should use the following commands with the environment activated.

> pip3 install ipykernel
> python3 -m ipykernel install --user --name=venv_name

The virtual environment can then be selected using the kernel menu in the toolbar.


## Docker

We provide a Docker configuration for your convenience. Begin by installing Docker on your system. You can run the tool as:

```
# docker build -t fossil .
# docker run -it fossil bash
```

You are now inside the container. `/project` contains all these files and Ubuntu has all the requirements pre-installed.

