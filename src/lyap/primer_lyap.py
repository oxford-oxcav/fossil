import sympy as sp
import numpy as np
import copy
import torch
import z3

from src.shared.Primer import Primer
from src.shared.system import NonlinearSystem
from experiments.benchmarks.domain_fcns import * 
from src.lyap.cegis_lyap import Cegis as Cegis_lyap
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType
from src.shared.utils import Timeout, FailedSynthesis
from src.shared.cegis_values import PrimerLyapConfig, CegisStateKeys, CegisConfig

class PrimerLyap(Primer):
    
    def __init__(self, f, **kw):
        self.dynamics = NonlinearSystem(f, True)
        self.shift = torch.zeros((1, self.dynamics.dimension))
        self.batch_size         = kw.get(PrimerLyapConfig.BATCH_SIZE.k, PrimerLyapConfig.BATCH_SIZE.v)
        self.outer_radius       = kw.get(PrimerLyapConfig.R.k, PrimerLyapConfig.R.v)
        self.inner_radius       = kw.get(PrimerLyapConfig.INNER_RADIUS.k, PrimerLyapConfig.INNER_RADIUS.v)
        self.interactive_domain = kw.get(PrimerLyapConfig.INTERACTIVE_DOMAIN.k, PrimerLyapConfig.INTERACTIVE_DOMAIN.v)
        self.positive_domain    = kw.get(PrimerLyapConfig.POSITIVE_DOMAIN.k, PrimerLyapConfig.POSITIVE_DOMAIN.v)
        self.seedbomb_handle    = kw.get(PrimerLyapConfig.SEEDBOMB.k, PrimerLyapConfig.SEEDBOMB.v)
        self.cegis_parameters   = kw.get(PrimerLyapConfig.CEGIS_PARAMETERS.k, PrimerLyapConfig.CEGIS_PARAMETERS.v)

    def get(self): 
        """
        :return V_n: numerical form of Lyap function
        :return V_v: symbolic form of Lyap function
        """
        self.get_shift()
        if self.seedbomb_handle:
            state = self.seedbomb()
        if self.interactive_domain:
            state = self.interactive_cegis()
        else:
            state, f_learner = self.run_cegis()
        
        if not state[CegisStateKeys.found]:
            raise FailedSynthesis('Function could not be synthesised.')
 
        learner = state[CegisStateKeys.net]

        def V_n(x):
            """
            :param x: iterable of shape (N, dimension) (torch tensor recommended)
            :return V, Vdot: torch tensors
            """
            if isinstance(x, torch.Tensor):
                x = x.reshape(-1, self.dynamics.dimension)
            else:
                x = torch.tensor(x).reshape(-1, self.dynamics.dimension)
            phi = x - self.shift
            xdot = torch.stack(f_learner(phi.T)).T
            V, Vdot, _ = learner.numerical_net(phi, xdot, state[CegisStateKeys.factors])
            return V, Vdot

        V_v = self.shift_symbolic_formula(state[CegisStateKeys.V], state[CegisStateKeys.x_v_map])

        return V_n, V_v

    def get_shift(self):
        """
        Selects the equilibrium for Lyapunov analysis (through user if necessary) and determines
        shift needed to move it to the origin.
        """

        if len(self.dynamics.stable_equilibria) > 1:
            index = self.get_user_choice() -1 
            eqbm = self.dynamics.stable_equilibria[index]
            print("Chosen Equilibrium  Point:\n {} ".format(eqbm))
            self.shift = torch.tensor([float(x) for x in eqbm]).T

        elif len(self.dynamics.stable_equilibria) == 1:
            print("Single Equilibrium point found: \n {}".format(self.dynamics.stable_equilibria))
            eqbm = self.dynamics.stable_equilibria[0]
            #self.shift_symbolic = np.array([z3.RealVal(x) for x in eqbm])
            self.shift = torch.tensor([float(x) for x in eqbm]).T
            self.sympy_shift = eqbm

        else:
            print("Error, no stable equilibria found.")
            choice = input("If this is an error, you may enter a proposed equilibrium point. y/N: ")
            if choice.lower() == "y":
                eqbm = self.get_user_eqbm()
                if eqbm is not None:
                    self.shift = torch.tensor([float(x) for x in eqbm]).T

    def change_domain(self, learner):
        """
        Offers user an interactive domain update.
        :param learner: NN learner object from CEGIS loop
        """
        print("CEGIS has been unable to find a Lyapunov function. Trying again with a smaller domain?\n")
        print("Recommended domain: hypersphere of radius {}".format(learner.closest_unsat))
        print("y/N?: ")
        if input() == "y":
            self.outer_radius = learner.closest_unsat
        else:
            self.interactive_domain = False
    
    def get_user_choice(self):
        """
        returns choice: integer from 1,...,N denoting which equilibrium user has chosen for Lyapunov analysis
        """

        print("\nMultiple stable equilibrium points found: \n")
        print({i+1:self.dynamics.stable_equilibria[i] for i in range(len(self.dynamics.stable_equilibria))})
        print("\n Please select equilibrium point for Lyapunov analysis. Enter integer from 1 to {}".format(len(self.dynamics.stable_equilibria)))
        invalid_input = True
        while invalid_input:
            choice = input()
            invalid_input = self.check_input(choice)

        return int(choice)

    def check_input(self, choice):
        """
        :param choice: string
        :return boolean: True if input is invalid, False if valid
        """
        try:
            return int(choice) not in range(1, len(self.dynamics.stable_equilibria) + 1)
        except ValueError:
            print("\nValue Error, please enter an integer. \n")
            return True
      
    def get_user_eqbm(self):
        """
        :return eqbm: list of sympy numbers, eqbm point
        """
        eqbm = sp.sympify(input("Please enter equilibrium point in form [x_0*, x_1*, ..., x_n*]"))
        if self.validate_eqbm_input(eqbm):
            return eqbm
        else:
            print("Invalid equilibrium point.")
            raise FailedSynthesis("No stable equilibria to perform analysis on.")

    def validate_eqbm_input(self, eqbm):
        """
        :param eqbm: sympified input of equilibiurm point in form [x_0*, x_1*, ..., x_n*]
        :return bool: True if equilibrium is valid (f(x*) = 0) else False
        """
        zero = [sp.core.numbers.Zero() for iii in range(self.dynamics.dimension)]
        xdot = self.dynamics.f_substitute(eqbm)

        return xdot == zero

    def interactive_cegis(self):
        """
        Loops through CEGIS until user no longer wants to update the domain.
        :return state: dict, CEGIS state dictionary
        :return f_learner: function that evaluates xdot of system
        """
        sat = False
        while (not sat and self.interactive_domain):
            state, f_learner = self.run_cegis()
            sat = state[CegisStateKeys.found]
            if not sat:
                self.change_domain(state[CegisStateKeys.net])
        return state, f_learner


    def run_cegis(self):
        """
        :return state: dict, CEGIS state dictionary
        :return f_learner: function that evaluates xdot of system
        """
        activations = self.cegis_parameters.get(PrimerLyapConfig.ACTIVATIONS.k, PrimerLyapConfig.ACTIVATIONS.v)
        n_hidden_neurons = self.cegis_parameters.get(PrimerLyapConfig.NEURONS.k, PrimerLyapConfig.NEURONS.v)

        learner_type = LearnerType.NN
        verifier_type = self.cegis_parameters.get(CegisConfig.VERIFIER.k, CegisConfig.VERIFIER.v)
        self.check_verifier(verifier_type)
        params = {CegisConfig.N_VARS.k:self.dynamics.dimension, CegisConfig.SYSTEM.k: self.system,
        CegisConfig.ACTIVATION.k: activations, CegisConfig.N_HIDDEN_NEURONS.k:n_hidden_neurons, 
        CegisConfig.INNER_RADIUS.k:self.inner_radius, CegisConfig.OUTER_RADIUS.k:self.outer_radius,
        CegisConfig.VERIFIER.k:verifier_type, CegisConfig.LEARNER.k:learner_type}
        self.cegis_parameters.update(params)
        c = Cegis_lyap(**self.cegis_parameters)
        state, x, f_learner, iters = c.solve()

        return state, f_learner
        
    def evaluate_dynamics(self, point):
        """
        :param choice: n-d data point as iterable
        :return f(point): (shifted) dynamical system evaluated at point 
        """
        if point.dtype=="O":
            # I think this shift needs to be done symbolically but not sure how to convert sympy number
            # to either z3 or dreal rational depending on verifier
            point = point + np.array(self.shift.reshape(point.shape))
        else:
            point = point + self.shift.unsqueeze(1)
        return self.dynamics.evaluate_f(point)

    def system(self, functions, inner=0.0, outer=10.0):
        _And = functions["And"]
        batch_size = self.batch_size
        def f(_, v):
            return self.evaluate_dynamics(v)
        
        def XD(_, v):
            if self.positive_domain:
                return _And(_And([v_i > 0 for v_i in v]),
                    sum([v_i ** 2 for v_i in v]) <= self.outer_radius**2,
                    self.inner_radius < sum([v_i ** 2 for v_i in v]))
            else:
                return _And(sum([v_i ** 2 for v_i in v]) <= self.outer_radius**2, self.inner_radius < sum([v_i ** 2 for v_i in v]))
        
        def SD():
            #Did not realise these were limited to 3D. TODO:
            origin = tuple([0 for iii in range(self.dynamics.dimension)])
            if self.positive_domain:
                return slice_nd_init_data(origin, self.outer_radius**2, batch_size)
            else:
                return round_init_data(origin, self.outer_radius **2, self.batch_size)
            return

        return f, XD, SD()

    def shift_symbolic_formula(self, V, x_v_map):
        """
        :param V: symbolic Lyapunov function, either from sympy, Z3 or dReal
        :param x_v_map: verifier variables from CEGIS 
        :return V: symbolic Lyapunov function shifted back according to equilibrium point
        """
        shift = np.array(self.shift)
        if isinstance(V, sp.Expr):
            s = {self.dynamics.x[i]: (self.dynamics.x[i] - shift[i]) for i in range(self.dynamics.dimension)}
            V = V.subs(s)
            return V
        if isinstance(V, z3.ArithRef):
            s = [(x,(x-shift[i])) for i, x in enumerate(x_v_map.values())]
            V = z3.substitute(V, s)
            return V
        else:
            s = {x:(x-shift[i]) for i, x in enumerate(x_v_map.values())}
            V = V.Substitute(s)
            return V