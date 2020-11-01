import sympy as sp
import numpy as np
import copy
import torch
from functools import partial

from src.shared.Primer import Primer
from src.shared.system import NonlinearSystem
from experiments.benchmarks.domain_fcns import * 
from experiments.benchmarks.benchmarks_bc import inf_bounds, inf_bounds_n
from src.barrier.cegis_barrier import Cegis as Cegis_barrier
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType
from src.shared.utils import Timeout
from src.shared.cegis_values import PrimerBarrierConfig, CegisStateKeys, CegisConfig

class PrimerBarrier(Primer):

    def __init__(self, f, xd, xi, xu, **kw):
        if not callable(f):
            # f is a list of sympy dynamics and Primer must generate the system and domains
            self.dynamics   = NonlinearSystem(f, False)
            self.dimension  = self.dynamics.dimension
            self.sxd, self.sxi, self.sxu = xd, xi, xu
        else:
            # f is directly a 'system' function of the form in the benchmarks
            self.dynamics = f
            self.dimension = kw.pop(PrimerBarrierConfig.DIMENSION.k, PrimerBarrierConfig.DIMENSION.v)
            if self.dimension is None:
                raise TypeError('Parameter DIMENSION must be passed if f is in the form of a python function.')

        self.seedbomb_handle  = kw.get(PrimerBarrierConfig.SEEDBOMB.k, PrimerBarrierConfig.SEEDBOMB.v)
        self.batch_size       = kw.get(PrimerBarrierConfig.BATCH_SIZE.k, PrimerBarrierConfig.BATCH_SIZE.v)
        self.cegis_parameters = kw.get(PrimerBarrierConfig.CEGIS_PARAMETERS.k, PrimerBarrierConfig.CEGIS_PARAMETERS.v)

    def get(self):
        """
        :return B_n: numerical form of Barrier function
        :return B_v: symbolic form of Barrier function
        """
        if self.seedbomb_handle:
            state, f_learner = self.seedbomb()
        else:
            state, f_learner = self.run_cegis()

        def B_n(x):
            """
            :param x: iterable of shape (N, dimension) (torch tensor recommended)
            :return B, Bdot:
            """
            if isinstance(x, torch.Tensor):
                x = x.reshape(-1, self.dynamics.dimension)
            else:
                x = torch.tensor(x).reshape(-1, self.dynamics.dimension)
            xdot = torch.stack(f_learner(x.T)).T
            B, Bdot, _ = state[CegisStateKeys.net].numerical_net(x, xdot)
            return B, Bdot
        
        B_v = state[CegisStateKeys.B]
        return B_n, B_v

    def run_cegis(self):
        """
        :return state: dict,  cegis state dictionary
        :return f_learner: function that evaluates xdot of system
        """
        if callable(self.dynamics):
            system = partial(self.dynamics, self.batch_size)
            dimension = self.dimension
        else:
            system = self.system
            dimension = self.dynamics.dimension
            
        activations = self.cegis_parameters.get(PrimerBarrierConfig.ACTIVATIONS.k, PrimerBarrierConfig.ACTIVATIONS.v)
        n_hidden_neurons = self.cegis_parameters.get(PrimerBarrierConfig.NEURONS.k, PrimerBarrierConfig.NEURONS.v)
        verifier_type = self.cegis_parameters.get(CegisConfig.VERIFIER.k, CegisConfig.VERIFIER.v)
        self.check_verifier(verifier_type)
        params = {CegisConfig.N_VARS.k:dimension, CegisConfig.SYSTEM.k: system, CegisConfig.ACTIVATION.k: activations,
        CegisConfig.N_HIDDEN_NEURONS.k:n_hidden_neurons, CegisConfig.VERIFIER.k:verifier_type, CegisConfig.LEARNER.k: CegisConfig.LEARNER.v}
        self.cegis_parameters.update(params)
        c = Cegis_barrier(**self.cegis_parameters)
                  
        state, x, f_learner, iters = c.solve()
        return state, f_learner

    def system(self, functions, inner=0.0, outer=10.0):
        _And = functions["And"]
        _Or  = functions["Or"]
        bounds = inf_bounds_n(self.dynamics.dimension)
        
        def recursive_AND(exp):
            if len(exp) == 1:
                return _And(exp[0])
            else:
                return _And(exp[0],recursive_AND(exp[1:]))

        def f(_,v):
            return self.dynamics.evaluate_f(v)
        
        def XD(_, v):
            return self.sxd.generate_domain(v, _And)

        def XI(_, v):
            return self.sxi.generate_domain(v, _And)

        def XU(_, v):
            return self.sxu.generate_domain(v, _And)      
        
        def SD():
            return self.sxd.generate_data(self.batch_size)
        
        def SI():
            return self.sxd.generate_data(self.batch_size)

        def SU():
            return self.sxd.generate_data(self.batch_size)

        return f, XD, XI, XU, SD(), SI(), SU(), bounds
