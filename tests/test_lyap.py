import torch
import timeit
import sympy as sp
from lyap.cegis import Cegis
from lyap.utils import compute_equilibria, check_real_solutions, dict_to_array
from tests.benchmarks.benchmarks_lyap import *
from shared.activations import ActivationType
from shared.consts import VerifierType, LearnerType
from functools import partial
from lyap.learner.sympy_solver import *


def main():
    n_vars = 2
    batch_size = 500

    system = partial(benchmark_3, batch_size)

    # compute equilibria
    f, _, __ = system(SympySolver.solver_fncts())
    f_sp = partial(f, SympySolver.solver_fncts())
    x_sp = [sp.Symbol('x%d' % i) for i in range(n_vars)]
    equilibria = compute_equilibria(f_sp(np.array(x_sp).reshape(1, 2)))
    real_eq = check_real_solutions(equilibria, x_sp)
    real_eq = dict_to_array(real_eq, n_vars)

    # define domain constraints
    outer_radius = 10
    inner_radius = 0.1

    # define NN parameters
    activations = [ActivationType.LINEAR]
    n_hidden_neurons = [5] * len(activations)

    learner_type = LearnerType.NN
    verifier_type = VerifierType.Z3

    """
    with factors = 'quadratic'
    the candidate Lyap is (x-eq0)^2 * ... * (x-eqN)^2 * NN(x)
    by passing factors = 'linear'
    have (x-eq0) * ... * (x-eqN) * NN(x)
    """
    factors = 'quadratic'

    start = timeit.default_timer()
    c = Cegis(n_vars, system, learner_type, activations, n_hidden_neurons,
              verifier_type, inner_radius, outer_radius,
              factors=factors, eq=real_eq, sp_handle=True)
    c.solve()
    stop = timeit.default_timer()
    print('Elapsed Time: {}'.format(stop-start))


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
