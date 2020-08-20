import traceback
import timeit
from lyap.cegis_lyap import Cegis as Cegis_for_lyap
from barrier.cegis_barrier import Cegis as Cegis_for_bc
from shared.activations import ActivationType
from shared.consts import VerifierType, LearnerType
from functools import partial


def lyap_synthesis(benchmark, n_vars):

    batch_size=500
    system = partial(benchmark, batch_size=batch_size)

    # compute equilibria
    # f, _, __ = system(functions=SympySolver.solver_fncts())
    # f_sp = partial(f, SympySolver.solver_fncts())
    # x_sp = [sp.Symbol('x%d' % i) for i in range(n_vars)]
    # equilibria = compute_equilibria(f_sp(np.array(x_sp)), x_sp)
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
    factors = None

    start = timeit.default_timer()
    c = Cegis_for_lyap(n_vars, system, learner_type, activations, n_hidden_neurons,
              verifier_type, inner_radius, outer_radius,
              factors=factors, eq=None, sp_handle=True)
    c.solve()
    stop = timeit.default_timer()
    print('Elapsed Time: {}'.format(stop-start))


def barrier_synthesis(benchmark, n_vars):

    MIN_TO_SEC = 60
    batch_size = 500
    system = partial(benchmark, batch_size)
    activations = [ActivationType.LINEAR, ActivationType.LIN_SQUARE_CUBIC, ActivationType.LINEAR]
    hidden_neurons = [2] * len(activations)
    try:
        start = timeit.default_timer()
        c = Cegis_for_bc(n_vars, LearnerType.NN, VerifierType.Z3, activations, system, hidden_neurons,
                  sp_simplify=True, cegis_time=30 * MIN_TO_SEC)
        _, found, _ = c.solve()
        end = timeit.default_timer()

        print('Elapsed Time: {}'.format(end - start))
        print("Found? {}".format(found))
    except Exception as _:
        print(traceback.format_exc())

