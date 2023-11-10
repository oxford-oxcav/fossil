# Copyright (c) 2023, Alessandro Abate, Alec Edwards, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

from typing import Any
from functools import partial

import torch
import numpy as np
import sympy as sp
import z3
import dreal
from cvc5 import pythonic as cvpy
from scipy import linalg
from matplotlib import pyplot as plt

from fossil import parser
from fossil import logger
from fossil.activations import activation
from fossil.activations_symbolic import activation_sym
from fossil.consts import Z3_FNCS, DREAL_FNCS, SP_FNCS, VerifierType, CVC5_FNCS
from fossil.utils import vprint, contains_object

ctrl_log = logger.Logger.setup_logger(__name__)


class DynamicalModel:
    def __init__(self) -> None:
        self.fncs = None

    def f(self, v):
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return self._f_torch(v)
        elif contains_object(v, dreal.Variable):
            self.fncs = DREAL_FNCS
            return self.f_smt(v)
        elif contains_object(v, z3.ArithRef):
            self.fncs = Z3_FNCS
            return self.f_smt(v)
        elif contains_object(v, cvpy.ArithRef):
            self.fncs = CVC5_FNCS
            return self.f_smt(v)
        elif contains_object(v, sp.Expr):
            self.fncs = SP_FNCS
            return self.f_smt(v)
        # Changed this so object is now pickleable, as long as self.fncs is None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.f(*args, **kwds)

    def f_torch(self, v: torch.Tensor) -> list:
        """Returns the output of the model as a list of torch tensors

        Args:
            v (torch.Tensor): (N_data, n_vars)

        Returns:
            list: length n_vars, each element is a torch tensor of shape (N_data, 1)
        """
        raise NotImplementedError

    def _f_torch(self, v: torch.Tensor) -> torch.Tensor:
        """Internal function to stack the output of f_torch, so that users do not have to import torch"""
        return torch.stack(self.f_torch(v)).T

    def f_smt(self, v):
        raise NotImplementedError

    def parameters(self):
        """Get learnable parameters of the model"""
        return ()

    def check_similarity(self):
        """
        Checks over a small number of data points that the learner & verifier funcs are the same.
        If false, does not create the object
        """
        x = np.random.rand(5, self.n_vars)
        f_torch = self.f_torch(torch.tensor(x)).detach().numpy()
        self.fncs = self.math_fncs
        f_smt = np.array([self.f_smt(xi) for xi in x])
        return np.allclose(f_torch, f_smt)

    def plot(self, ax=None, xrange=[-3, 3], yrange=[-3, 3]):
        ax = plt.gca() or ax
        xx = np.linspace(xrange[0], xrange[1], 50)
        yy = np.linspace(yrange[0], yrange[1], 50)
        XX, YY = np.meshgrid(xx, yy)
        dx, dy = (
            self._f_torch(
                torch.stack(
                    [torch.tensor(XX).ravel(), torch.tensor(YY).ravel()]
                ).T.float()
            )
            .detach()
            .numpy()
            .T
        )
        # color = np.sqrt((np.hypot(dx, dy)))
        dx = dx.reshape(XX.shape)
        dy = dy.reshape(YY.shape)
        # color = color.reshape(XX.shape)
        ax.set_ylim(xrange)
        ax.set_xlim(yrange)
        plt.streamplot(
            XX,
            YY,
            dx,
            dy,
            linewidth=0.8,
            density=1.5,
            arrowstyle="fancy",
            arrowsize=1.5,
            color="tab:gray",
        )
        return ax

    def to_latex(self):
        x = sp.symbols(",".join(("x" + str(i) for i in range(self.n_vars))))
        return sp.latex(self.f(x))

    def to_sympy(self):
        x = sp.symbols(",".join(("x" + str(i) for i in range(self.n_vars))))
        return self.f(x)

    def clean(self):
        """Prepare object for pickling"""
        self.fncs = None


class ControllableDynamicalModel:
    """Combine with a GeneralController to create a closed-loop model"""

    def __init__(self) -> None:
        self.fncs = None
        self.parameters = ()

    def f(self, v, u):
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return self._f_torch(v, u)
        elif contains_object(v, dreal.Variable):
            self.fncs = DREAL_FNCS
            return self.f_smt(v, u)
        elif contains_object(v, z3.ArithRef):
            self.fncs = Z3_FNCS
            return self.f_smt(v, u)
        elif contains_object(v, cvpy.ArithRef):
            self.fncs = CVC5_FNCS
            return self.f_smt(v, u)
        elif contains_object(v, sp.Expr):
            self.fncs = SP_FNCS
            return self.f_smt(v, u)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.f(*args, **kwds)

    def f_torch(self, v: torch.Tensor, u: torch.Tensor) -> list:
        """Returns the output of the model as a list of torch tensors

        Args:
            v (torch.Tensor): (N_data, n_vars)

        Returns:
            list: length n_vars, each element is a torch tensor of shape (N_data, 1)
        """
        raise NotImplementedError

    def _f_torch(self, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Internal function to stack the output of f_torch, so that users do not have to import torch"""
        return torch.stack(self.f_torch(v, u)).T

    def f_smt(self, v, u):
        raise NotImplementedError

    def clean(self):
        """Prepare object for pickling"""
        self.fncs = None

    def to_latex(self):
        # pass symbols to latex
        x = sp.symbols(["x" + str(i) for i in range(self.n_vars)])
        u = sp.symbols(["u" + str(i) for i in range(self.n_u)])
        return sp.latex(self.f(x, u))


class _ParsedDynamicalModel(DynamicalModel):
    """A CTModel that is parsed from a list of strings"""

    def __init__(self, f: list[str], verifier: VerifierType) -> None:
        super().__init__()
        self.n_vars = len(f)
        self.verifier = verifier
        self.f_num = self.get_numerical_f(f)
        self.f_sym = self.get_symbolic_f(f)

    def get_numerical_f(self, f):
        return parser.parse_dynamical_system_to_numpy(f)

    def get_symbolic_f(self, f):
        if self.verifier == VerifierType.DREAL:
            p = parser.DrealParser()
        elif self.verifier == VerifierType.Z3:
            p = parser.Z3Parser()
        elif self.verifier == VerifierType.CVC5:
            p = parser.CVC5Parser()
        else:
            raise ValueError(
                "Verifier {} not supported from command line".format(self.verifier)
            )

        return p.parse_dynamical_system(f)

    def f_torch(self, v):
        return [fi(v.T) for fi in self.f_num]

    def f_smt(self, v):
        return [fi(v) for fi in self.f_sym]


class _ParsedControllableDynamicalModel(ControllableDynamicalModel):
    """A ControllableCTModel that is parsed from a list of strings"""

    def __init__(self, f: list[str], verifier: VerifierType) -> None:
        super().__init__()
        self.n_vars = len(f)
        self.n_u = None  # Unknown at this point
        self.verifier = verifier
        self.f_num = self.get_numerical_f(f)
        self.f_sym = self.get_symbolic_f(f)

    def get_numerical_f(self, f):
        return parser.parse_dynamical_system_to_numpy(f)

    def get_symbolic_f(self, f):
        if self.verifier == VerifierType.DREAL:
            p = parser.DrealParser()
        elif self.verifier == VerifierType.Z3:
            p = parser.Z3Parser()
        elif self.verifier == VerifierType.CVC5:
            p = parser.CVC5Parser()
        else:
            raise ValueError(
                "Verifier {} not supported from command line".format(self.verifier)
            )

        res = p.parse_dynamical_system(f)
        self.n_u = len(p.us)
        assert self.n_u > 0
        assert self.n_vars == len(p.xs)
        return res

    def f_torch(self, v, u):
        return [fi(v.T, u.T) for fi in self.f_num]

    def f_smt(self, v, u):
        return [fi(v, u) for fi in self.f_sym]


class GeneralController(torch.nn.Module):
    """
    GeneralController generalises the Controller module in the dimensionality definition.
    The 'learn' method is empty, because it doesnt "learn" per se, but always in order to
    make the Lyapunov conditions valid.
    Might merge the Base and Generalised Controller in the future...
    """

    def __init__(self, inputs, output, layers, activations) -> None:
        super(GeneralController, self).__init__()
        self.inp = inputs
        self.out = output
        self.acts = activations
        self.layers = []

        n_prev = self.inp
        for k, n_neurons in enumerate(layers):
            layer = torch.nn.Linear(n_prev, n_neurons, bias=False)
            self.register_parameter("W" + str(k), layer.weight)
            # self.register_parameter("b" + str(k), layer.bias)
            self.layers.append(layer)
            n_prev = n_neurons
        output = torch.nn.Linear(n_prev, self.out, bias=False)
        self.register_parameter("W" + str(k + 1), output.weight)
        # self.register_parameter("b" + str(k + 1), layer.bias)
        self.layers.append(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for act, layer in zip(self.acts, self.layers):
            z = layer(y)
            y = activation(act, z)
        z = self.layers[-1](y)
        return z

    def reset_parameters(self):
        for layer in self.layers:
            torch.nn.init.uniform(layer.weight, -1, 1)

    def learn(self, x: torch.Tensor, f_open: torch.Tensor, optimizer: torch.optim):
        pass

    def to_symbolic(self, x, verbose=False) -> list:
        y = np.atleast_2d(x).T
        rounding = 5

        for act, layer in zip(self.acts, self.layers):
            W = layer.weight.detach().numpy().round(rounding)
            # b = layer.bias.detach().numpy().round(rounding)
            z = np.atleast_2d(W @ y)  # + b
            y = activation_sym(act, z)
        W = self.layers[-1].weight.detach().numpy().round(rounding)
        # b = self.layers[-1].bias.detach().numpy().round(rounding)
        z = W @ y  # + b

        ctrl_log.info(f"Controller: \n{z}")

        return z.reshape(-1).tolist()


# supports not-full-rank-affine and not-affine systems
class GeneralClosedLoopModel(DynamicalModel):
    """Class for synthesising a controller alongside a certificate with the same loss function.
    Combine a ControllableCTModel with a GeneralController"""

    def __init__(
        self, f_open: ControllableDynamicalModel, controller: GeneralController
    ) -> None:
        """Combine a controllable model with a general controller.

        Args:
            f_open (ControllableCTModel): open loop
            controller (GeneralController): control net
        """
        super().__init__()
        self.open_loop = f_open
        self.controller = controller
        self.n_vars = f_open.n_vars
        self.n_u = f_open.n_u
        # self.reset_controller()

    def f_torch(self, v):
        u = self.controller(v)
        return self.open_loop(v, u)

    def _f_torch(self, v):
        return self.f_torch(v)

    def f_smt(self, v):
        fc = self.controller.to_symbolic(v)
        fo = self.open_loop(v, fc)
        return [fo[i] for i in range(len(fo))]

    def parameters(self):
        """Get learnable parameters of the model"""
        return self.controller.parameters()

    def clean(self):
        """Prepare object for pickling"""
        self.fncs = None
        self.open_loop.clean()

    def reset_controller(self):
        while not self.check_stabilty():
            self.controller.reset_parameters()

    # def check_stabilty(self):
    #     lin = Lineariser(self)
    #     A = lin.linearise()
    #     E = EigenCalculator(A)
    #     print("Eigenvalues of linearised system: ", E.eigs)
    #     # self.plot()
    #     # plt.show()
    #     return E.is_stable()

    @classmethod
    def prepare_from_open(cls, f_open: ControllableDynamicalModel):
        """Prepare a closed loop model from an open loop model, which then must be called with a controller."""
        return partial(cls, f_open)


def nonzero_loss1(S: torch.Tensor, Sdot: torch.Tensor):
    # penalise zero control not near the origin
    gamma = torch.norm(S, p=2, dim=1) ** 2
    return (-gamma * torch.norm(Sdot, p=2, dim=1)).mean()


def nonzero_loss1b(S: torch.Tensor, Sdot: torch.Tensor):
    # penalise zero control not near the origin
    # gamma = torch.norm(S, p=2, dim=1) ** 2
    relu = torch.relu
    bias = 0.1
    nonzero_penalty = relu(-torch.norm(Sdot, p=2, dim=1) + bias)
    return (nonzero_penalty).mean()


def nonzero_loss2(S: torch.Tensor, Sdot: torch.Tensor):
    # penalise difference between control in each dimension
    # \sum_i=/=j |u_i - u_j|
    l = 0
    for i in range(S.shape[1] - 1):
        j = i + 1
        l += (torch.abs(Sdot[:, i]) / torch.abs(Sdot[:, j]) - 1).mean()
    return l


def ridge_reg(S: torch.Tensor, Sdot: torch.Tensor):
    return torch.norm(Sdot, p=2, dim=1).mean()


def ridge_reg_param(W: list[torch.Tensor]):
    s = 0
    s = s + torch.mean(torch.stack([torch.norm(Wi, p=2) for Wi in W]))
    return s


def cosine_reg(S: torch.Tensor, Sdot: torch.Tensor):
    cosine = torch.nn.CosineSimilarity(dim=1)
    loss = cosine(Sdot, S)
    return loss.mean()


def saturated_cosine_reg(S: torch.Tensor, Sdot: torch.Tensor):
    cosine = torch.nn.CosineSimilarity(dim=1)
    relu = torch.nn.Softplus()
    loss = relu(cosine(Sdot, S))
    return loss.mean()


def saturated_cosine_reg2(S: torch.Tensor, Sdot: torch.Tensor):
    cosine = torch.nn.CosineSimilarity(dim=1)
    relu = torch.nn.ReLU()
    loss = relu(cosine(Sdot, S) + 0.1)
    return loss.mean()


class Lineariser:
    """Linearises the model around the origin, which is assumed to be the equilibrium point"""

    def __init__(self, model):
        self.model = model
        self.n_vars = self.model.n_vars
        try:
            self.n_u = self.model.n_u
        except AttributeError:
            self.n_u = 0
        self.x = [sp.Symbol("x" + str(i), real=True) for i in range(self.n_vars)]
        self.u = [sp.Symbol("u" + str(i), real=True) for i in range(self.n_u)]
        self._check_zero()
        # zero = torch.tensor([0.0] * self.n_vars).unsqueeze(0)
        # print(self.model(zero))

    def get_model(self) -> list[sp.Expr]:
        """Returns the model as a list of sympy expressions"""
        try:
            return self.model(self.x)
        except TypeError:
            return self.model(self.x, self.u)

    def _check_zero(self):
        """Checks if the model is zero at the origin"""
        f = sp.Matrix(self.get_model())
        f = f.subs([(x, 0) for x in self.x])
        f = f.subs([(u, 0) for u in self.u])
        f0 = [fi != 0 for fi in f]
        if any(f0):
            raise ValueError("Model is not zero at the origin")

    def get_jacobian(self) -> sp.Matrix:
        """Returns the jacobian of the model"""
        J = sp.Matrix(self.get_model()).jacobian(self.x)
        return J

    def linearise(self) -> np.ndarray:
        """Linearises the model around the origin, returning the A matrix"""
        J = self.get_jacobian()
        J_0 = J.subs([(x, 0) for x in self.x])
        return np.array(J_0).astype(np.float64)


class EigenCalculator:
    """Class to help calculate and analyse the eigenvalues of a linear model"""

    def __init__(self, A) -> None:
        self.A = A
        self.dim = A.shape[0]
        self.eigs = self.get_eigs()

    def get_eigs(self) -> np.ndarray:
        """Returns the eigenvalues of the model"""
        return np.linalg.eigvals(self.A)

    def get_real_parts(self) -> list[float]:
        """Returns the real parts of the eigenvalues"""
        return [e.real for e in self.eigs]

    def is_stable(self) -> bool:
        """Checks if the model is stable"""
        return all([e.real < 0 for e in self.eigs])

    def is_sufficiently_stable(self) -> bool:
        """Checks if the poles are well places"""
        return all([e.real < -1 for e in self.eigs])

    def get_worst_pole(self) -> float:
        "Returns least stable pole"
        return max(self.get_real_parts())


class LQR:
    """Generates the LQR controller for a linear model"""

    def __init__(self, A, B=None, Q=None, R=None):
        """Initialises the LQR controller

        Args:
            A (np.ndarray): state matrix of the linear model
            B (np.ndarray): input matrix of the linear model
            Q (np.ndarray, optional): Defaults to Identity.
            R (np.ndarray, optional): Defaults to Identity.
        """
        self.A = A
        self.n_vars = A.shape[0]
        self.B = B
        assert B.shape[0] == self.n_vars
        self.n_u = self.B.shape[1]
        self.Q = Q if Q is not None else np.eye(self.n_vars)
        self.R = R if R is not None else np.eye(self.n_u)

    def solve(self) -> np.ndarray:
        """Solves the LQR Riccati equation, returning the K matrix"""
        P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import dreal
    from fossil import consts

    f = ["-x1 + u0", "x0-x1"]
    x0, x1 = dreal.Variable("x0"), dreal.Variable("x1")
    open_model = _ParsedControllableDynamicalModel(f, VerifierType.DREAL)
    model = GeneralClosedLoopModel(
        open_model, GeneralController(2, 1, [5], [consts.ActivationType.LINEAR])
    )
    x0 = dreal.Variable("x0")
    x1 = dreal.Variable("x1")
    x = np.array([x0, x1]).reshape(-1, 1)
    x = [x0, x1]
    xt = torch.randn(2, 1).T
    print(model(x))
    print(model(xt))
