import sympy as sp
from experiments.benchmarks.create_bcmk import create_benchmark_for_lyap
from functools import partial
from experiments.init_synth import lyap_synthesis


def synthesis_rm_lyap(input_rm):

    # get the flow from the RM
    flow = input_rm.get_flow()
    # if flow has len 1 -> ODE
    if len(flow) == 1:
        # extract vars from system in input
        # NOTA: this only works if vars are x0, x1, etc.
        # generalisation: get the string from free_symbols and make z3 vars
        sp_vars = input_rm.get_vars()
        dyna = sp.lambdify(sp_vars, flow[0].output)
    # if len(flow) > 1 -> Hybrid System. How to handle?
    else:
        sp_vars = input_rm.get_vars()
        guards = input_rm.get_guards()
        # assume mutually exclusive guards (*very* easy case)
        dyna = []
        dyna += [sp.lambdify(sp_vars, d) for d in input_rm.get_dynamics()]

    benchmark = partial(create_benchmark_for_lyap, dynamics=input_rm)
    lyap_synthesis(benchmark=benchmark, n_vars=len(sp_vars))
