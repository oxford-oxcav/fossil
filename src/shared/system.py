import sympy as sp
import numpy as np
import copy
import torch
try:
    import dreal
except:
    dreal = None
    
from src.shared.utils import Timeout

class LinearSystem():
    def __init__(self, A):
        self.A = A
        self.dimension = len(A[0])
        self.x = np.array([sp.symbols("x%d" % i) for i in range(self.dimension)])
        self.f = self.get_func_lambda()

    def get_func_from_A(self,):
        """
        returns xdot: Sympy expression representing the linear system
        """
        xdot = self.A @ self.x
        return xdot

    def get_func_lambda(self):
        """
        returns f: python function which evaluates linear system 
        """
        xdot = self.get_func_from_A()
        f = sp.lambdify((self.x), xdot, "numpy")
        return f

    def evaluate_f(self, point):
        """
        param choice: n-d data point as iterable
        returns f(point): dynamical system evaluated at point 
        """
        return self.f(*point)

class NonlinearSystem():
    def __init__(self, f, lyap=True):
        """
        :param f: list representing each dimensions dynamics, with each element i is f_i(x0,x1,x2...,xn)
        :param lyap: bool, mode defining lyapunov or barrier function operation
        """
        self.f = f
        self.poly = self.check_poly()
        self.dimension = len(f)
        self.x = [sp.Symbol("x%d" % i, real=True) for i in range(self.dimension)]
        self.system_lambda = self.get_system_lambda()

        if not self.poly:
            self.dreal_lambda = self.get_dreal_lambda()
            self.sympy_lambda = self.get_sympy_lambda()
        if lyap:
            if not self.poly:
                raise ValueError("Non-polynomial dynamics not supported for Lyapunov analysis.")
            self.equilibria = self.find_equilibria()
            self.jacobian = self.get_Jacobian()
            self.stable_equilibria  = []
            self.unstable_equilibria = []
            self.sort_equilibria()

    def get_system_lambda(self):
        """
        :return f: function which evaluates system 
        """
        f = sp.lambdify(self.x, self.f, modules=[{"sin":torch.sin, "exp": torch.exp, "cos":torch.cos}, "numpy"])
        return f

    def get_dreal_lambda(self):
        """
        :return f: function which evaluates system using dreal functions
        """
        f = sp.lambdify(self.x, self.f, modules=[{"sin":dreal.sin, "exp": dreal.exp, "cos":dreal.cos}, "numpy"])
        return f

    def get_sympy_lambda(self):
        """
        :return f: function which evaluates system that using sympy functions
        """
        f = sp.lambdify(self.x, self.f, modules=[{"sin":sp.sin, "exp": sp.exp, "cos":sp.cos}, "numpy"])
        return f

    def evaluate_f(self, point):
        """
        :param point: n-d data point as iterable
        :return f(point): dynamical system evaluated at point 
        """
        if dreal:
            if isinstance(point[0], dreal.Variable):
                return self.dreal_lambda(*point)
            elif isinstance(point[0], sp.Expr):
                return self.sympy_lambda(*point)
            else:
                return self.system_lambda(*point)
        else:
            return self.system_lambda(*point)

    def get_Jacobian(self):
        """
        :return J: Jacobion of system, numpy object matrix with Sympy expressions for each entry
        """

        J  = np.zeros((self.dimension, self.dimension), dtype=object)
        for jjj, state in enumerate(self.x):
            for iii, fun in enumerate(self.f):
                J[iii,jjj] = sp.diff(fun, state)

        return J

    def evaluate_Jacobian(self, point):
        """
        :param point: list representing n-d point at which to evaluate the Jacobian J
        :return J_x*: np array of Jacobian evaluated at point  
        """
        J_x = copy.deepcopy(
            self.jacobian
        )
        for iii, df in enumerate(J_x):
            for jjj, df_dx in enumerate(df):
                J_x[iii,jjj] = float(
                    df_dx.subs({x: p for (x, p) in zip(self.x, point)})
                )
        return np.array(J_x, dtype=float)

    def find_equilibria(self):
        """
        :return real_equilibria: list of equilibrium points for system
        """

        try:
            with Timeout(seconds=180):
                eqbm = sp.nonlinsolve(self.f, self.x,)
        except TimeoutError:
            eqbm = []
        except AttributeError:
            eqbm = sp.nonlinsolve(self.f, self.x,)
        
        real_equilibria = self.get_real_solutions(eqbm.args)

        return real_equilibria

    def get_real_solutions(self, eqbm_set):
        """
        :param eqbm_set: list of equilibrium points (in complex domain)
        :return real_equilibria: list of equilibrium points for system (in R^n)
        """
        real_equilibria = []
        for eqbm in eqbm_set:
            real_Flag = True
            for number in eqbm:
                if not number.is_real:
                    real_Flag = False
            if real_Flag:
                #eqbm = tuple([float(x) for x in eqbm])
                real_equilibria.append(eqbm)
        return real_equilibria

    def check_stability(self, J='0', eqbm=None):
        """
        :param J: Jacobian of dynamical system, possibly evaluated at specifc equilibrium point
        :param eqbm: equilibrium point to evaluate Jacobian at if not already evaluated.
        :return bool: True if all eigenvalues have real part <= 0, else False.
        """
        if type(J) is str:
            J = self.evaluate_Jacobian(eqbm)
        V,_ = np.linalg.eig(J)
        return all(np.real(V) <= 0)

    def sort_equilibria(self):
        for eqbm in self.equilibria:
            J = self.evaluate_Jacobian(eqbm)
            if self.check_stability(J=J):
                self.stable_equilibria.append(eqbm)
            else:
                self.unstable_equilibria.append(eqbm)

    def f_substitute(self, point):
        """
        :param point: iterable, point at which to symbolically evaluate f
        :return f(point): symbolic evaluation (by substitution) of self.f at point
        """
        substitutions = {x: p for (x, p) in zip(self.x, point)}
        return [(f_i.subs(substitutions)) for f_i in self.f]

    def check_poly(self):
        """
        :return bool: False if system has any non-polynomial parts (eg exp, sin)
        """
        return not any([expression.has(sp.exp, sp.sin, sp.cos) for expression in self.f])
    