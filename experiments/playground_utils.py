import sympy as sp
import torch
from z3 import ArithRef, simplify

from src.shared.Primer import Primer
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType
from experiments.benchmarks.domain_fcns import Rectangle, Sphere

from src.shared.system import NonlinearSystem
exp, sin, cos = sp.exp, sp.sin, sp.cos

def print_f(function):
    """
    Attempts to print symbolic function:
    :param function: symbolic function of type sympy.exp, z3.ArithRef or dreal
    """
    if isinstance(function, sp.exp):
        f = sp.simplify(function)
        print(f)
    if isinstance(function, ArithRef):
        f = simplify(function)
        print(f)
    else:
        print(function)

def plot_f(function):
    raise NotImplementedError
    

def initialise_states(N):
    """
    :param N: int, number of states to initialise
    :return states: tuple of states as symp vars x0,...,xN
    """
    states = " ".join(["x%d" %i for i in range(N)])
    v = sp.symbols(states, real=True)
    return v

def synthesise(f, mode, **kwargs):
    p = Primer.create_Primer(f, mode=mode, **kwargs)
    return p.get()
