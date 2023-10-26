import sys
import inspect
from typing import Any
from functools import partial

import dreal
import numpy as np
import sympy as sp
import torch
import z3
from matplotlib import pyplot as plt

from src import control
from src.utils import contains_object

Z3_FNCS = {
    "And": z3.And,
    "Or": z3.Or,
    "If": z3.If,
}
DREAL_FNCS = {
    "sin": dreal.sin,
    "cos": dreal.cos,
    "exp": dreal.exp,
    "And": dreal.And,
    "Or": dreal.Or,
    "If": dreal.if_then_else,
    "Not": dreal.Not,
}
MATH_FNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
}

SP_FNCS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "exp": sp.exp,
}


class CTModel:
    def __init__(self) -> None:
        self.fncs = None

    def f(self, v):
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return self.f_torch(v)
        elif contains_object(v, dreal.Variable):
            self.fncs = DREAL_FNCS
            return self.f_smt(v)
        elif contains_object(v, z3.ArithRef):
            self.fncs = Z3_FNCS
            return self.f_smt(v)
        elif contains_object(v, sp.Expr):
            self.fncs = SP_FNCS
            return self.f_smt(v)
        # Changed this so object is now pickleable, as long as self.fncs is None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.f(*args, **kwds)

    def f_torch(self, v):
        raise NotImplementedError

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
            self.f_torch(
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


class ControllableCTModel:
    """Combine with a GeneralController to create a closed-loop model"""

    def __init__(self) -> None:
        self.fncs = None
        self.parameters = ()

    def f(self, v, u):
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return self.f_torch(v, u)
        elif contains_object(v, dreal.Variable):
            self.fncs = DREAL_FNCS
            return self.f_smt(v, u)
        elif contains_object(v, z3.ArithRef):
            self.fncs = Z3_FNCS
            return self.f_smt(v, u)
        elif contains_object(v, sp.Expr):
            self.fncs = SP_FNCS
            return self.f_smt(v, u)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.f(*args, **kwds)

    def f_torch(self, v):
        raise NotImplementedError

    def f_smt(self, v):
        raise NotImplementedError

    def clean(self):
        """Prepare object for pickling"""
        self.fncs = None

    def to_latex(self):
        x = sp.symbols(",".join(("x" + str(i) for i in range(self.n_vars))))
        u = sp.symbols(",".join(("u" + str(i) for i in range(self.n_u))) + ",")
        return sp.latex(self.f(x, u))


class _PreTrainedModel(CTModel):
    """Unused class for training a closed-loop model with a NN controller and then certifying it.

    Uses a separate loss function for the controller and the certificate."""

    def __init__(self, f_open: CTModel, controller: control.StabilityCT) -> None:
        super().__init__()
        self.open_loop = f_open
        self.controller = controller
        self.parameters = ()

    def f_torch(self, v):
        return self.open_loop(v) + self.controller(v).detach()

    def f_smt(self, v):
        fo = self.open_loop(v)
        fc = self.controller.to_symbolic(v)
        return [fo[i] + fc[i, 0] for i in range(len(fo))]


# supports not-full-rank-affine and not-affine systems
class GeneralClosedLoopModel(CTModel):
    """Class for synthesising a controller alongside a certificate with the same loss function.
    Combine a ControllableCTModel with a GeneralController"""

    def __init__(
        self, f_open: ControllableCTModel, controller: control.GeneralController
    ) -> None:
        """Combine a controllable model with a general controller.

        Args:
            f_open (ControllableCTModel): open loop
            controller (control.GeneralController): control net
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

    def check_stabilty(self):
        lin = control.Lineariser(self)
        A = lin.linearise()
        E = control.EigenCalculator(A)
        print("Eigenvalues of linearised system: ", E.eigs)
        # self.plot()
        # plt.show()
        return E.is_stable()

    @classmethod
    def prepare_from_open(cls, f_open: ControllableCTModel):
        """Prepare a closed loop model from an open loop model, which then must be called with a controller."""
        return partial(cls, f_open)


class Eulerised:
    """
    Create discrete time model from continuous time model using the Euler method:
    y' = y + h * f(x)
    """

    def __init__(self, f, h) -> None:
        self.h = h
        self.fc = f
        # self.name = f.__name__()

    def f(self, v):
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return v + self.h * self.fc(v)
        else:
            f_v = self.fc(v)
            return [v0 + self.h * v1 for v0, v1 in zip(v, f_v)]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.f(*args, **kwds)


############################################
# LYAPUNOV BENCHMARKS
############################################


class Linear0(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-x - y, x]).T

    def f_smt(self, v):
        x, y = v
        return [-x - y, x]


class NonPoly0(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-x + x * y, -y]).T

    def f_smt(self, v):
        x, y = v
        return [-x + x * y, -y]


class NonPoly1(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-x + 2 * x**2 * y, -y]).T

    def f_smt(self, v):
        x, y = v
        return [-x + 2 * x**2 * y, -y]


class NonPoly2(CTModel):
    n_vars = 3

    def f_torch(self, v):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        return torch.stack([-x, -2 * y + 0.1 * x * y**2 + z, -z - 1.5 * y]).T

    def f_smt(self, v):
        x, y, z = v
        return [-x, -2 * y + 0.1 * x * y**2 + z, -z - 1.5 * y]


class NonPoly3(CTModel):
    n_vars = 3

    def f_torch(self, v):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        return torch.stack([-3 * x - 0.1 * x * y**3, -y + z, -z]).T

    def f_smt(self, v):
        x, y, z = v
        return [-3 * x - 0.1 * x * y**3, -y + z, -z]


class Benchmark0(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-x, -y]).T

    def f_smt(self, v):
        x, y = v
        return [-x, -y]


class Benchmark1(ControllableCTModel):
    n_u = 2
    n_vars = 2

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2 = u[:, 0], u[:, 1]
        return torch.stack([x + y + u1, -y - x + u2]).T

    def f_smt(self, v, u):
        x, y = v
        u1, u2 = u
        return [x + y + u1, -y - x + u2]


class Benchmark2(ControllableCTModel):
    n_vars = 2
    n_u = 3

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
        return torch.stack([x + y + u1 - u2, y + 2.0 * x + u3]).T

    def f_smt(self, v, u):
        x, y = v
        u1, u2, u3 = u
        return [x + y + u1 - u2, y + 2.0 * x + u3]


class BenchmarkDT1(ControllableCTModel):
    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2 = u[:, 0], u[:, 1]
        return torch.stack([2.0 * x + u1, 2.0 * y + u2]).T

    def f_smt(self, v, u):
        x, y = v
        u1, u2 = u
        return [2.0 * x + u1, 2.0 * y + u2]


# POLY benchmarks
# this series comes from
# https://www.cs.colorado.edu/~srirams/papers/nolcos13.pdf
# srirams paper from 2013 (old-ish) but plenty of lyap fcns


class Poly1(CTModel):
    n_vars = 3

    def f_torch(self, v):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        return torch.stack(
            [-(x**3) - x * z**2, -y - x**2 * y, -z + 3 * x**2 * z - (3 * z)]
        ).T

    def f_smt(self, v):
        x, y, z = v
        return [-(x**3) - x * z**2, -y - x**2 * y, -z + 3 * x**2 * z - (3 * z)]


class Poly2(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-(x**3) + y, -x - y]).T

    def f_smt(self, v):
        x, y = v
        return [-(x**3) + y, -x - y]


class Poly3(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-(x**3) - y**2, x * y - y**3]).T

    def f_smt(self, v):
        x, y = v
        return [-(x**3) - y**2, x * y - y**3]


class Poly4(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack(
            [-x - 1.5 * x**2 * y**3, -(y**3) + 0.5 * x**3 * y**2]
        ).T

    def f_smt(self, v):
        x, y = v
        return [-x - 1.5 * x**2 * y**3, -(y**3) + 0.5 * x**3 * y**2]


class Sriram4D(CTModel):
    n_vars = 4

    def f_torch(self, v):
        x1, x2, x3, x4 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        return torch.stack(
            [
                -x1 + x2**3 - 3 * x3 * x4,
                -x1 - x2**3,
                x1 * x4 - x3,
                x1 * x3 - x4**3,
            ]
        ).T

    def f_smt(self, v):
        x1, x2, x3, x4 = v
        return [
            -x1 + x2**3 - 3 * x3 * x4,
            -x1 - x2**3,
            x1 * x4 - x3,
            x1 * x3 - x4**3,
        ]


class LinearDiscrete(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([0.5 * x - 0.5 * y, 0.5 * x]).T

    def f_smt(self, v):
        x, y = v
        return [0.5 * x - 0.5 * y, 0.5 * x]


class DoubleLinearDiscrete(CTModel):
    n_vars = 4

    def f_torch(self, v):
        x1, x2, x3, x4 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        return torch.stack(
            [0.5 * x1 - 0.5 * x2, 0.5 * x1, 0.5 * x3 - 0.5 * x4, 0.5 * x3]
        ).T

    def f_smt(self, v):
        x1, x2, x3, x4 = v
        return [0.5 * x1 - 0.5 * x2, 0.5 * x1, 0.5 * x3 - 0.5 * x4, 0.5 * x3]


class LinearDiscreteNVars(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([0.5 * v[i] for i in range(len(v))]).T

    def f_smt(self, v):
        x, y = v
        return [0.5 * v[i] for i in range(len(v))]


class NonLinearDiscrete(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([0.5 * x - 0.000001 * y**2, 0.5 * x * y]).T

    def f_smt(self, v):
        x, y = v
        return [0.5 * x - 0.000001 * y**2, 0.5 * x * y]


# class TwoDHybrid(CTModel):
#     n_vars = 2

#     def f_torch(self, v):
#         x0, x1 = v[:, 0], v[:, 1]
#         _condition = x1 >= 0
#         _negated_cond = x1 < 0
#         _then = -x1 - 0.5 * x0**3
#         _else = -x1 - x0**2 - 0.25 * x1**3
#         # _condition and _negated _condition are tensors of bool, act like 0 and 1
#         x1dot = _condition * _then + _negated_cond * _else

#         return torch.stack([-x0, x1dot]).T

#     def f_smt(self, v):
#         _If = self.fncs["If"]
#         x0, x1 = v
#         _then = -x1 - 0.5 * x0**3
#         _else = -x1 - x0**2 - 0.25 * x1**3
#         _cond = x1 >= 0
#         return [-x0, _If(_cond, _then, _else)]

# class TwoD_Hybrid(CTModel):
#     n_vars = 2

#     def f_torch(self, v):
#         x0, x1 = v[:, 0], v[:, 1]
#         _then = -x0 - 0.5 * x0**3
#         _else = x0 - 0.25 * x1**2
#         _cond = x0 >= 0
#         return torch.stack([x1, torch.where(_cond, _then, _else)]).T

#     def f_smt(self, v):
#         x0, x1 = v
#         If = self.fncs["If"]
#         _then = -x0 - 0.5 * x0**3
#         _else = x0 - 0.25 * x1**2
#         _cond = x0 >= 0
#         return [x1, If(_cond, _then, _else)]


############################################
# BARRIER BENCHMARKS
############################################


class Barr1(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([y + 2 * x * y, -x - y**2 + 2 * x**2]).T

    def f_smt(self, v):
        x, y = v
        return [y + 2 * x * y, -x - y**2 + 2 * x**2]


class Barr2(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([torch.exp(-x) + y - 1, -(torch.sin(x) ** 2)]).T

    def f_smt(self, v):
        sin = self.fncs["sin"]
        exp = self.fncs["exp"]
        x, y = v
        return [exp(-x) + y - 1, -((sin(x)) ** 2)]


class Barr3(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([y, -x - y + 1 / 3 * x**3]).T

    def f_smt(self, v):
        x, y = v
        return [y, -x - y + 1 / 3 * x**3]


class ObstacleAvoidance(CTModel):
    n_vars = 3

    def f_torch(self, v):
        x, y, phi = v[:, 0], v[:, 1], v[:, 2]
        velo = 1
        return torch.stack(
            [
                velo * torch.sin(phi),
                velo * torch.cos(phi),
                -torch.sin(phi)
                + 3
                * (x * torch.sin(phi) + y * torch.cos(phi))
                / (0.5 + x**2 + y**2),
            ]
        ).T

    def f_smt(self, v):
        x, y, phi = v
        velo = 1
        sin = self.fncs["sin"]
        cos = self.fncs["cos"]
        return [
            velo * sin(phi),
            velo * cos(phi),
            -sin(phi) + 3 * (x * sin(phi) + y * cos(phi)) / (0.5 + x**2 + y**2),
        ]


class HighOrd4(CTModel):
    n_vars = 4

    def f_torch(self, v):
        x0, x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        return torch.stack(
            [x1, x2, x3, -3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0]
        ).T

    def f_smt(self, v):
        x0, x1, x2, x3 = v
        return [x1, x2, x3, -3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0]


class HighOrd6(CTModel):
    n_vars = 6

    def f_torch(self, v):
        x0, x1, x2, x3, x4, x5 = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5]
        return torch.stack(
            [
                x1,
                x2,
                x3,
                x4,
                x5,
                -800 * x5 - 2273 * x4 - 3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0,
            ]
        ).T

    def f_smt(self, v):
        x0, x1, x2, x3, x4, x5 = v
        return [
            x1,
            x2,
            x3,
            x4,
            x5,
            -800 * x5 - 2273 * x4 - 3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0,
        ]


class HighOrd8(CTModel):
    n_vars = 8

    def f_torch(self, v):
        x0, x1, x2, x3, x4, x5, x6, x7 = (
            v[:, 0],
            v[:, 1],
            v[:, 2],
            v[:, 3],
            v[:, 4],
            v[:, 5],
            v[:, 6],
            v[:, 7],
        )
        return torch.stack(
            [
                x1,
                x2,
                x3,
                x4,
                x5,
                x6,
                x7,
                -20 * x7
                - 170 * x6
                - 800 * x5
                - 2273 * x4
                - 3980 * x3
                - 4180 * x2
                - 2400 * x1
                - 576 * x0,
            ]
        ).T

    def f_smt(self, v):
        x0, x1, x2, x3, x4, x5, x6, x7 = v
        return [
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            -20 * x7
            - 170 * x6
            - 800 * x5
            - 2273 * x4
            - 3980 * x3
            - 4180 * x2
            - 2400 * x1
            - 576 * x0,
        ]


class UnstableLinear(CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-2 * x - y, 0.6 * y]).T

    def f_smt(self, v):
        x, y = v
        return [-2 * x - y, 0.6 * y]


class Car(CTModel):
    n_vars = 3

    def f_torch(self, v):
        x, y, omega = v[:, 0], v[:, 1], v[:, 2]
        return torch.stack([torch.cos(omega), torch.sin(omega), omega]).T

    def f_smt(self, v):
        x, y, omega = v
        sin = self.fncs["sin"]
        cos = self.fncs["cos"]
        return [cos(omega), sin(omega), omega]


class InvertedPendulum(ControllableCTModel):
    n_vars = 2
    n_u = 2

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2 = u[:, 0], u[:, 1]

        G = 9.81  # gravity
        L = 0.5  # length of the pole
        m = 0.15  # ball mass
        b = 0.1  # friction

        return torch.stack(
            [y + u1, u2 + (m * G * L * torch.sin(x) - b * y) / (m * L**2)]
        ).T

    def f_smt(self, v, u):
        x, y = v
        u1, u2 = u
        sin = self.fncs["sin"]
        cos = self.fncs["cos"]
        # Dynamics
        G = 9.81  # gravity
        L = 0.5  # length of the pole
        m = 0.15  # ball mass
        b = 0.1  # friction

        return [y + u1, u2 + (m * G * L * sin(x) - b * y) / (m * L**2)]


class InvertedPendulumLQR(CTModel):
    n_vars = 2
    K = np.array([[7.21, 1.34], [1.34, 0.33]])

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        u1 = -self.K[0, 0] * x - self.K[0, 1] * y
        u2 = -self.K[1, 0] * x - self.K[1, 1] * y

        G = 9.81  # gravity
        L = 0.5  # length of the pole
        m = 0.15  # ball mass
        b = 0.1  # friction

        return torch.stack(
            [y + u1, u2 + (m * G * L * torch.sin(x) - b * y) / (m * L**2)]
        ).T

    def f_smt(self, v):
        x, y = v
        u1 = -self.K[0, 0] * x - self.K[0, 1] * y
        u2 = -self.K[1, 0] * x - self.K[1, 1] * y
        sin = self.fncs["sin"]
        cos = self.fncs["cos"]
        # Dynamics
        G = 9.81  # gravity
        L = 0.5  # length of the pole
        m = 0.15  # ball mass
        b = 0.1  # friction

        return [y + u1, u2 + (m * G * L * sin(x) - b * y) / (m * L**2)]


class LorenzSystem(ControllableCTModel):
    n_vars = 3
    n_u = 3

    def f_torch(self, v, u):
        x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2]
        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]

        sigma = 10.0  # related to Prandtl number,
        r = 28.0  # related to Rayleigh number,
        b = 8.0 / 3.0  # geometric factor (Weisstein, 2002).

        return torch.stack(
            [-sigma * (x1 - x2) + u1, r * x1 - x2 - x1 * x3 + u2, x1 * x2 - b * x3 + u3]
        ).T

    def f_smt(self, v, u):
        x1, x2, x3 = v
        u1, u2, u3 = u

        # Dynamics
        sigma = 10.0  # related to Prandtl number,
        r = 28.0  # related to Rayleigh number,
        b = 8.0 / 3.0  # geometric factor (Weisstein, 2002).

        return [
            -sigma * (x1 - x2) + u1,
            r * x1 - x2 - x1 * x3 + u2,
            x1 * x2 - b * x3 + u3,
        ]


class CtrlCar(ControllableCTModel):
    n_vars = 3

    def f_torch(self, v, u):
        x, y, omega = v[:, 0], v[:, 1], v[:, 2]
        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
        return torch.stack([torch.cos(omega) + u1, torch.sin(omega) + u2, omega + u3]).T

    def f_smt(self, v, u):
        x, y, omega = v
        u1, u2, u3 = u
        sin = self.fncs["sin"]
        cos = self.fncs["cos"]
        return [cos(omega) + u1, sin(omega) + u2, omega + u3]


# from Tedrake's lecture notes
class Quadrotor2d(ControllableCTModel):
    n_vars = 6
    n_u = 2

    def __init__(self):
        super().__init__()
        # parameters based on [Bouadi, Bouchoucha, Tadjine, 2007]
        self.length = 0.25  # length of rotor arm
        self.mass = 0.486  # mass of quadrotor
        self.inertia = 0.00383  # moment of inertia
        self.gravity = 9.81  # gravity

    def f_torch(self, v, u):
        u1, u2 = u[:, 0], u[:, 1]
        # with respect to the original paper, we define
        # w1 = u1+u2
        # w2 = u1-u2
        q = v[:, :3]
        qdot = v[:, 3:]
        qddot = torch.vstack(
            [
                -torch.sin(q[:, 2]) / self.mass * u1,
                torch.cos(q[:, 2]) / self.mass * u1 - self.gravity,
                self.length / self.inertia * u2,
            ]
        ).T

        return torch.hstack([qdot, qddot])

    def f_smt(self, v, u):
        sin = self.fncs["sin"]
        cos = self.fncs["cos"]
        u1, u2 = u
        # with respect to the original paper, we define
        # w1 = u1+u2
        # w2 = u1-u2
        q = v[:3]
        qdot = v[3:]
        qddot = [
            -sin(q[2]) / self.mass * u1,
            cos(q[2]) / self.mass * u1 - self.gravity,
            self.length / self.inertia * u2,
        ]

        return [*qdot, *qddot]


# from Tedrake's lecture notes
class LinearSatellite(ControllableCTModel):
    n_vars = 5
    n_u = 3

    def __init__(self):
        super().__init__()
        # parameters based on [Bouadi, Bouchoucha, Tadjine, 2007]
        self.mass = 1.0  # mass of quadrotor
        self.gravity = 9.81  # gravity
        # data taken from
        # https://github.com/MIT-REALM/neural_clbf/
        MU = 3.986e14
        a = 42164e3
        self.n = MU / (a**3)

    def f_torch(self, v, u):
        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
        # with respect to the original paper, we define
        # w1 = u1+u2
        # w2 = u1-u2
        q = v[:, :3]
        qdot = v[:, 3:]
        qddot = torch.vstack(
            [
                3.0 * self.n**2 * q[:, 0]
                - 2.0 * self.n * qdot[:, 1]
                + u1 / self.mass,
                -2.0 * self.n * qdot[:, 0] + u2 / self.mass,
                -self.n**2 * q[:, 2] + u3 / self.mass,
            ]
        ).T

        return torch.hstack([qdot, qddot])

    def f_smt(self, v, u):
        u1, u2, u3 = u
        # with respect to the original paper, we define
        # w1 = u1+u2
        # w2 = u1-u2
        q = v[:3]
        qdot = v[3:]
        qddot = [
            3.0 * self.n**2 * q[0] - 2.0 * self.n * qdot[1] + u1 / self.mass,
            -2.0 * self.n * qdot[0] + u2 / self.mass,
            -self.n**2 * q[2] + u3 / self.mass,
        ]

        return [*qdot, *qddot]


class CtrlObstacleAvoidance(ControllableCTModel):
    n_vars = 3
    n_u = 1

    def f_torch(self, v, u):
        x, y, phi = v[:, 0], v[:, 1], v[:, 2]
        u1 = u[:, 0]
        velo = 1
        return torch.stack(
            [
                velo * torch.sin(phi),
                velo * torch.cos(phi),
                -torch.sin(phi) + u1,
            ]
        ).T

    def f_smt(self, v, u):
        x, y, phi = v
        (u1,) = u
        velo = 1
        sin = self.fncs["sin"]
        cos = self.fncs["cos"]
        return [
            velo * sin(phi),
            velo * cos(phi),
            -sin(phi) + u1,
        ]


class Identity(ControllableCTModel):
    n_vars = 2
    n_u = 2

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2 = u[:, 0], u[:, 1]
        return torch.stack([x + u1, y + u2]).T

    def f_smt(self, v, u):
        x, y = v
        u1, u2 = u
        return [x + u1, y + u2]


class DTAhmadi(ControllableCTModel):
    # from Non-monotonic Lyapunov Functions
    # for Stability of Discrete Time Nonlinear and Switched Systems
    # Amir Ali Ahmadi and Pablo A. Parrilo, CDC 2008.
    n_vars = 2
    n_u = 2

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2 = u[:, 0], u[:, 1]
        return torch.stack([0.3 * x + u1, -x + 0.5 * y + 7.0 / 18.0 * x**2 + u2]).T

    def f_smt(self, v, u):
        x, y = v
        u1, u2 = u
        return [0.3 * x + u1, -x + 0.5 * y + 7.0 / 18.0 * x**2 + u2]


### Benchmarks taken from RSWS work of Verdier, Mazo


class Linear1(ControllableCTModel):
    n_vars = 2
    n_u = 1

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1 = u[:, 0]
        return torch.stack([y, -x + u1]).T

    def f_smt(self, v, u):
        x, y = v
        (u1,) = u
        return [y, -x + u1]


class Linear1LQR(CTModel):
    """Linear1 with LQR controller"""

    n_vars = 2
    K = [0.414, 1.352]

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        u1 = -(self.K[0] * x + self.K[1] * y)
        return torch.stack([y, -x + u1]).T

    def f_smt(self, v):
        x, y = v
        u1 = -(self.K[0] * x + self.K[1] * y)
        return [y, -x + u1]


class SecondOrder(ControllableCTModel):
    n_vars = 2
    n_u = 1

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1 = u[:, 0]
        return torch.stack([y - x**3, u1]).T

    def f_smt(self, v, u):
        x, y = v
        (u1,) = u
        return [y - x**3, u1]


class SecondOrderLQR(CTModel):
    n_vars = 2
    K = [1.0, 1.73]

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        u1 = -(self.K[0] * x + self.K[1] * y)
        return torch.stack([y - x**3, u1]).T

    def f_smt(self, v):
        x, y = v
        u1 = -(self.K[0] * x + self.K[1] * y)
        return [y - x**3, u1]


class ThirdOrder(ControllableCTModel):
    n_vars = 3
    n_u = 1

    def f_torch(self, v, u):
        x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2]
        u1 = u[:, 0]
        return torch.stack(
            [-10 * x1 + 10 * x2 + u1, 28 * x1 - x2 - x1 * x3, x1 * x2 - 8 / 3 * x3]
        ).T

    def f_smt(self, v, u):
        x1, x2, x3 = v
        (u1,) = u
        return [-10 * x1 + 10 * x2 + u1, 28 * x1 - x2 - x1 * x3, x1 * x2 - 8 / 3 * x3]


class ThirdOrderLQR(CTModel):
    n_vars = 3
    K = [23.71, 18.49, 0.0]

    def f_torch(self, v):
        x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2]
        u1 = -(self.K[0] * x1 + self.K[1] * x2 + self.K[2] * x3)
        return torch.stack(
            [-10 * x1 + 10 * x2 + u1, 28 * x1 - x2 - x1 * x3, x1 * x2 - 8 / 3 * x3]
        ).T

    def f_smt(self, v):
        x1, x2, x3 = v
        u1 = -(self.K[0] * x1 + self.K[1] * x2 + self.K[2] * x3)
        return [-10 * x1 + 10 * x2 + u1, 28 * x1 - x2 - x1 * x3, x1 * x2 - 8 / 3 * x3]


class LoktaVolterra(CTModel):
    n_vars = 2

    def __init__(self) -> None:
        super().__init__()

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([0.6 * x - x * y, -0.6 * y + x * y]).T

    def f_smt(self, v):
        x, y = v
        return [0.6 * x - x * y, -0.6 * y + x * y]


class VanDerPol(CTModel):
    n_vars = 2

    def __init__(self) -> None:
        super().__init__()

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([y, -0.5 * (1 - x**2) * y - x]).T

    def f_smt(self, v):
        x, y = v
        return [y, -0.5 * (1 - x**2) * y - x]


class SheModel(CTModel):
    n_vars = 6

    def f_torch(self, v):
        x1, x2, x3, x4, x5, x6 = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5]
        return torch.stack(
            [
                x2 * x4 - x1**3,
                -3 * x1 * x4 - x2**3,
                -x3 - 3 * x1 * x4**3,
                -x4 + x1 * x3,
                -x5 + x6**3,
                -x5 - x6 + x3**4,
            ]
        ).T

    def f_smt(self, v):
        x1, x2, x3, x4, x5, x6 = v
        return [
            x2 * x4 - x1**3,
            -3 * x1 * x4 - x2**3,
            -x3 - 3 * x1 * x4**3,
            -x4 + x1 * x3,
            -x5 + x6**3,
            -x5 - x6 + x3**4,
        ]


class PapaPrajna6(CTModel):
    n_vars = 6

    def f_torch(self, v):
        x1, x2, x3, x4, x5, x6 = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5]
        return torch.stack(
            [
                -(x1**3) + 4 * x2**3 - 6 * x3 * x4,
                -x1 - x2 + x5**3,
                x1 * x4 - x3 + x4 * x6,
                x1 * x3 + x3 * x6 - x4**3,
                -2 * x2**3 - x5 + x6,
                -3 * x3 * x4 - x5**3 - x6,
            ]
        ).T

    def f_smt(self, v):
        x1, x2, x3, x4, x5, x6 = v
        return [
            -(x1**3) + 4 * x2**3 - 6 * x3 * x4,
            -x1 - x2 + x5**3,
            x1 * x4 - x3 + x4 * x6,
            x1 * x3 + x3 * x6 - x4**3,
            -2 * x2**3 - x5 + x6,
            -3 * x3 * x4 - x5**3 - x6,
        ]


def read_model(model_string: str) -> CTModel:
    """Read model from string and return model object"""
    clsmembers = inspect.getmembers(
        sys.modules[__name__],
        lambda x: inspect.isclass(x) and (x.__module__ == __name__),
    )
    for M, _ in clsmembers:
        if M == model_string:
            try:
                model = getattr(sys.modules[__name__], M)()
            except (AttributeError, TypeError):
                # Attribute error: M is not correct name
                # TypeError: M is in namespace but not a CTModel - eg someone has tried to pass dreal
                raise ValueError(f"{model_string} is not a CTModel")
            return model
    raise ValueError(f"{model_string} is not a CTModel")


def _all_models_to_latex() -> str:
    """Return a string containing latex for all models"""
    latex = ""
    clsmembers = inspect.getmembers(
        sys.modules[__name__],
        lambda x: inspect.isclass(x) and (x.__module__ == __name__),
    )
    for name, model in clsmembers:
        try:
            m = model()
            if isinstance(m, CTModel):
                latex += name + "\n"
                latex += m.to_latex() + "\n\n"
        except (AttributeError, TypeError):
            pass
    return latex


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="Barr3")
    args = parser.parse_args()
    model = read_model(args.model)
    ax = model.plot()
    plt.show()
    # print(_all_models_to_latex())
