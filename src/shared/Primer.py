import sympy as sp
import numpy as np
import copy
import torch

from src.shared.system import NonlinearSystem
from experiments.benchmarks.domain_fcns import * 
from experiments.benchmarks.benchmarks_bc import inf_bounds, inf_bounds_n
from src.lyap.cegis_lyap import Cegis as Cegis_lyap
from src.barrier.cegis_barrier import Cegis as Cegis_barrier
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType
from src.shared.utils import Timeout, FailedSynthesis
from src.shared.cegis_values import CegisStateKeys


class Primer():

    @staticmethod
    def create_Primer(f, **kw):
        """
        Instantiate either a PrimerLyap or PrimerBarrier object, depending on keyword argument 'mode'.
        :param f: dynamical system as list of sympy expressions.
        """
        mode = kw.get("mode")
        if mode.lower() == "l":
            from src.lyap.primer_lyap import PrimerLyap
            return PrimerLyap(f, **kw)
        if mode.lower() == "b":
            xd = kw.get("XD")
            xi = kw.get("XI")
            xu = kw.get("XU") 
            from src.barrier.primer_barrier import PrimerBarrier
            return PrimerBarrier(f, xd, xi, xu, **kw)

    def seedbomb(self, time=8, max_attempts=500):
        """
        Call CEGIS repeatedly for short attempts with different random seeds
        :param time: int, time to allow CEGIS to run for in seconds
        :param max_attempts: int, number of CEGIS attempts before stopping
        :return state: dict, CEGIS state 
        :return f_learner: function that evaluates xdot of system 
        """
        sat = False
        attempts = 0
        while not sat and attempts < max_attempts:
            try:
                with Timeout(seconds=time):
                    state, f_learner = self.run_cegis()
                    sat = state[CegisStateKeys.found]
                    attempts += 1
            except TimeoutError:
                state, f_learner = [{CegisStateKeys.found:False}, None]
            except AttributeError:
                raise ValueError("Seedbomb functionality only available on UNIX systems.")
            else:
                return state, f_learner

    def check_verifier(self, verifier):
        """
        Validates verifier choice with dynamics and activations
        :param verifier: VerifierType 
        """
        if verifier == VerifierType.Z3:
            if not self.dynamics.poly:
                raise ValueError('Z3 not compatible with non-polynomial dynamics.')
            activations =  self.cegis_parameters.pop("ACTIVATIONS", [ActivationType.SQUARE])
            dreal_activations = [ActivationType.SIGMOID, ActivationType.TANH]
            if [i for i in dreal_activations if i in activations]:
                raise ValueError('Z3 not compatible with chosen activation functions.')

