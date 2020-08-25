import sympy as sp
import torch
from experiments.benchmarks.create_bcmk import create_benchmark_for_lyap, create_benchmark_for_barrier
from functools import partial
from tst.reg.init_synth import lyap_synthesis, barrier_synthesis


def main():

    input_sys = sp.simplify(input('Please enter the system: '))
    lyap_or_barrier = ''
    while lyap_or_barrier != 'l' and lyap_or_barrier != 'b':
        lyap_or_barrier = input('Lyapunov (l) or Barrier (b)? ')
        print('Please enter either l or b')

    lambda_f = sp.lambdify(input_sys.free_symbols, input_sys)
    # extract vars from system in input
    # NOTA: this only works if vars are x0, x1, etc.
    # generalisation: get the string from free_symbols and make z3 vars
    sp_vars = input_sys.free_symbols

    if lyap_or_barrier == 'l':
        benchmark = partial(create_benchmark_for_lyap, dynamics=lambda_f)
        lyap_synthesis(benchmark=benchmark, n_vars=len(sp_vars))
    else:
        benchmark = partial(create_benchmark_for_barrier, dynamics=lambda_f)
        barrier_synthesis(benchmark=benchmark, n_vars=len(sp_vars))


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
