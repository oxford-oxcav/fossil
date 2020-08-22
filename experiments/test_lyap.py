import torch
import timeit
from src.lyap.cegis_lyap import Cegis
from src.lyap.utils import compute_equilibria, check_real_solutions, dict_to_array
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType
from functools import partial
from src.lyap.learner.sympy_solver import *


def main():
    n_vars = 2
    batch_size = 500

    system = partial(benchmark_4, batch_size)

    # compute equilibria
    f, _, __ = system(SympySolver.solver_fncts())
    f_sp = partial(f, SympySolver.solver_fncts())
    x_sp = [sp.Symbol('x%d' % i) for i in range(n_vars)]
    # todo: computation of equilibria must be handled correctly
    # equilibria = compute_equilibria(f_sp(np.array(x_sp).reshape(1, 2)))
    # real_eq = check_real_solutions(equilibria, x_sp)
    # real_eq = dict_to_array(real_eq, n_vars)

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
              factors=factors, sp_handle=True)
    c.solve()
    stop = timeit.default_timer()
    print('Elapsed Time: {}'.format(stop-start))


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
