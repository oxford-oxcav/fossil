from typing import Any

import torch
import z3
import dreal
import numpy as np
from matplotlib import pyplot as plt

from src.shared.utils import contains_object
from src.shared import control


class CTModel:
    def __init__(self) -> None:
        if not self.check_similarity():
            raise RuntimeError("Model functions not properly defined")
        self.z3_fncs = {
            "And": z3.And,
            "Or": z3.Or,
            "If": z3.If,
        }
        self.dreal_fncs = {
            "sin": dreal.sin,
            "cos": dreal.cos,
            "exp": dreal.exp,
            "And": dreal.And,
            "Or": dreal.Or,
            "If": dreal.if_then_else,
            "Not": dreal.Not,
        }
        self.fncs = None
        self.parameters = ()

    def f(self, v):
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return self.f_torch(v)
        elif contains_object(v, dreal.Variable):
            self.fncs = self.dreal_fncs
            return self.f_smt(v)
        elif contains_object(v, z3.ArithRef):
            self.fncs = self.z3_fncs
            return self.f_smt(v)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.f(*args, **kwds)

    def f_torch(self, v):
        raise NotImplementedError

    def f_smt(self, v):
        raise NotImplementedError

    def check_similarity(self):
        """
        Checks over a small number of data points that the learner & verifier funcs are the same.
        If false, does not create the object
        Not implemented yet
        """
        return True

    def plot(self):
        xrange = [-3, 3]
        yrange = [-3, 3]
        ax = plt.gca()
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
        color = np.sqrt((np.hypot(dx, dy)))
        dx = dx.reshape(XX.shape)
        dy = dy.reshape(YY.shape)
        color = color.reshape(XX.shape)
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
            color=color,
        )
        plt.show()


class GeneralCTModel:
    def __init__(self) -> None:
        if not self.check_similarity():
            raise RuntimeError("Model functions not properly defined")
        self.z3_fncs = {
            "And": z3.And,
            "Or": z3.Or,
            "If": z3.If,
        }
        self.dreal_fncs = {
            "sin": dreal.sin,
            "cos": dreal.cos,
            "exp": dreal.exp,
            "And": dreal.And,
            "Or": dreal.Or,
            "If": dreal.if_then_else,
            "Not": dreal.Not,
        }
        self.fncs = None
        self.parameters = ()

    def f(self, v, u):
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return self.f_torch(v, u)
        elif contains_object(v, dreal.Variable):
            self.fncs = self.dreal_fncs
            return self.f_smt(v, u)
        elif contains_object(v, z3.ArithRef):
            self.fncs = self.z3_fncs
            return self.f_smt(v, u)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.f(*args, **kwds)

    def f_torch(self, v):
        raise NotImplementedError

    def f_smt(self, v):
        raise NotImplementedError

    def check_similarity(self):
        """
        Checks over a small number of data points that the learner & verifier funcs are the same.
        If false, does not create the object
        Not implemented yet
        """
        return True


class ClosedLoopModel(CTModel):
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
    def __init__(
        self, f_open: GeneralCTModel, controller: control.GeneralController
    ) -> None:
        super().__init__()
        self.open_loop = f_open
        self.controller = controller
        self.parameters = controller.parameters()

    def f_torch(self, v):
        u = self.controller(v)
        return self.open_loop(v, u)

    def f_smt(self, v):
        fc = self.controller.to_symbolic(v)
        fo = self.open_loop(v, fc)
        return [fo[i] for i in range(len(fo))]


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


class NonPoly0(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-x + x * y, -y]).T

    def f_smt(self, v):
        x, y = v
        return [-x + x * y, -y]


class NonPoly1(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-x + 2 * x ** 2 * y, -y]).T

    def f_smt(self, v):
        x, y = v
        return [-x + 2 * x ** 2 * y, -y]


class NonPoly2(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        return torch.stack([-x, -2 * y + 0.1 * x * y ** 2 + z, -z - 1.5 * y]).T

    def f_smt(self, v):
        x, y, z = v
        return [-x, -2 * y + 0.1 * x * y ** 2 + z, -z - 1.5 * y]


class NonPoly3(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        return torch.stack([-3 * x - 0.1 * x * y ** 3, -y + z, -z]).T

    def f_smt(self, v):
        x, y, z = v
        return [-3 * x - 0.1 * x * y ** 3, -y + z, -z]


# POLY benchmarks
# this series comes from
# https://www.cs.colorado.edu/~srirams/papers/nolcos13.pdf
# srirams paper from 2013 (old-ish) but plenty of lyap fcns


class Benchmark0(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-x, -y]).T

    def f_smt(self, v):
        x, y = v
        return [-x, -y]


class Benchmark1(GeneralCTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2 = u[:, 0], u[:, 1]
        return torch.stack([x + y + u1, -y - x + u2]).T

    def f_smt(self, v, u):
        x, y = v
        u1, u2 = u[0, 0], u[1, 0]
        return [x + y + u1, -y - x + u2]


class Benchmark2(GeneralCTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
        return torch.stack([x + y + u1 - u2, y + 2.0 * x + u3]).T

    def f_smt(self, v, u):
        x, y = v
        u1, u2, u3 = u
        return [x + y + u1 - u2, y + 2.0 * x + u3]


class BenchmarkDT1(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([2.0 * x, 2.0 * y]).T

    def f_smt(self, v):
        x, y = v
        return [2.0 * x, 2.0 * y]


class Poly1(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        return torch.stack(
            [-(x ** 3) - x * z ** 2, -y - x ** 2 * y, -z + 3 * x ** 2 * z - (3 * z)]
        ).T

    def f_smt(self, v):
        x, y, z = v
        return [-(x ** 3) - x * z ** 2, -y - x ** 2 * y, -z + 3 * x ** 2 * z - (3 * z)]


class Poly2(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-(x ** 3) + y, -x - y]).T

    def f_smt(self, v):
        x, y = v
        return [-(x ** 3) + y, -x - y]


class Poly3(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-(x ** 3) - y ** 2, x * y - y ** 3]).T

    def f_smt(self, v):
        x, y = v
        return [-(x ** 3) - y ** 2, x * y - y ** 3]


class Poly4(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack(
            [-x - 1.5 * x ** 2 * y ** 3, -(y ** 3) + 0.5 * x ** 3 * y ** 2]
        ).T

    def f_smt(self, v):
        x, y = v
        return [-x - 1.5 * x ** 2 * y ** 3, -(y ** 3) + 0.5 * x ** 3 * y ** 2]


class TwoDHybrid(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x0, x1 = v[:, 0], v[:, 1]
        _condition = x1 >= 0
        _negated_cond = x1 < 0
        _then = -x1 - 0.5 * x0 ** 3
        _else = -x1 - x0 ** 2 - 0.25 * x1 ** 3
        # _condition and _negated _condition are tensors of bool, act like 0 and 1
        x1dot = _condition * _then + _negated_cond * _else

        return torch.stack([-x0, x1dot]).T

    def f_smt(self, v):
        _If = self.fncs["If"]
        x0, x1 = v
        _then = -x1 - 0.5 * x0 ** 3
        _else = -x1 - x0 ** 2 - 0.25 * x1 ** 3
        _cond = x1 >= 0
        return [-x0, _If(_cond, _then, _else)]


class LinearDiscrete(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([0.5 * x - 0.5 * y, 0.5 * x]).T

    def f_smt(self, v):
        x, y = v
        return [0.5 * x - 0.5 * y, 0.5 * x]


class DoubleLinearDiscrete(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x1, x2, x3, x4 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        return torch.stack(
            [0.5 * x1 - 0.5 * x2, 0.5 * x1, 0.5 * x3 - 0.5 * x4, 0.5 * x3]
        ).T

    def f_smt(self, v):
        x1, x2, x3, x4 = v
        return [0.5 * x1 - 0.5 * x2, 0.5 * x1, 0.5 * x3 - 0.5 * x4, 0.5 * x3]


class LinearDiscreteNVars(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([0.5 * v[i] for i in range(len(v))]).T

    def f_smt(self, v):
        x, y = v
        return [0.5 * v[i] for i in range(len(v))]


class NonLinearDiscrete(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([0.5 * x - 0.000001 * y ** 2, 0.5 * x * y]).T

    def f_smt(self, v):
        x, y = v
        return [0.5 * x - 0.000001 * y ** 2, 0.5 * x * y]


class Hybrid2d(CTModel):
    def f_torch(self, v):
        x0, x1 = v[:, 0], v[:, 1]

        _then = -x1 - 0.5 * x0 ** 3
        _else = -x1 - x0 ** 2 - 0.25 * x1 ** 3
        _cond = x1 >= 0

        return torch.stack([-x0, torch.where(_cond, _then, _else)]).T

    def f_smt(self, v):
        If = self.fncs["If"]
        x0, x1 = v
        _then = -x1 - 0.5 * x0 ** 3
        _else = -x1 - x0 ** 2 - 0.25 * x1 ** 3
        _cond = x1 >= 0
        return [-x0, If(_cond, _then, _else)]


############################################
# BARRIER BENCHMARKS
############################################


class Barr1(CTModel):
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([y + 2 * x * y, -x - y ** 2 + 2 * x ** 2]).T

    def f_smt(self, v):
        x, y = v
        return [y + 2 * x * y, -x - y ** 2 + 2 * x ** 2]


class Barr2(CTModel):
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([torch.exp(-x) + y - 1, -(torch.sin(x) ** 2)]).T

    def f_smt(self, v):
        sin = self.fncs["sin"]
        exp = self.fncs["exp"]
        x, y = v
        return [exp(-x) + y - 1, -((sin(x)) ** 2)]


class Barr3(CTModel):
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([y, -x - y + 1 / 3 * x ** 3]).T

    def f_smt(self, v):
        x, y = v
        return [y, -x - y + 1 / 3 * x ** 3]


class TwoD_Hybrid(CTModel):
    def f_torch(self, v):
        x0, x1 = v[:, 0], v[:, 1]
        _then = -x0 - 0.5 * x0 ** 3
        _else = x0 - 0.25 * x1 ** 2
        _cond = x0 >= 0
        return torch.stack([x1, torch.where(_cond, _then, _else)]).T

    def f_smt(self, v):
        x0, x1 = v
        If = self.fncs["If"]
        _then = -x0 - 0.5 * x0 ** 3
        _else = x0 - 0.25 * x1 ** 2
        _cond = x0 >= 0
        return [x1, If(_cond, _then, _else)]


class ObstacleAvoidance(CTModel):
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
                / (0.5 + x ** 2 + y ** 2),
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
            -sin(phi) + 3 * (x * sin(phi) + y * cos(phi)) / (0.5 + x ** 2 + y ** 2),
        ]


class HighOrd4(CTModel):
    def f_torch(self, v):
        x0, x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        return torch.stack(
            [x1, x2, x3, -3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0]
        ).T

    def f_smt(self, v):
        x0, x1, x2, x3 = v
        return [x1, x2, x3, -3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0]


class HighOrd6(CTModel):
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
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-2 * x - y, 0.6 * y]).T

    def f_smt(self, v):
        x, y = v
        return [-2 * x - y, 0.6 * y]


class Car(CTModel):
    def f_torch(self, v):
        x, y, omega = v[:, 0], v[:, 1], v[:, 2]
        return torch.stack([torch.cos(omega), torch.sin(omega), omega]).T

    def f_smt(self, v):
        x, y, omega = v
        sin = self.fncs["sin"]
        cos = self.fncs["cos"]
        return [cos(omega), sin(omega), omega]


class InvertedPendulum(GeneralCTModel):
    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2 = u[:, 0], u[:, 1]

        G = 9.81  # gravity
        L = 0.5  # length of the pole
        m = 0.15  # ball mass
        b = 0.1  # friction

        return torch.stack(
            [y + u1, u2 + (m * G * L * torch.sin(x) - b * y) / (m * L ** 2)]
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

        return [y + u1, u2 + (m * G * L * sin(x) - b * y) / (m * L ** 2)]


# from Tedrake's lecture notes
class Quadrotor2d(GeneralCTModel):
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
        u1, u2 = u[0], u[1]
        # with respect to the original paper, we define
        # w1 = u1+u2
        # w2 = u1-u2
        q = v[:3]
        qdot = np.array(v[3:])
        qddot = np.array(
            [
                -sin(q[2]) / self.mass * u1,
                cos(q[2]) / self.mass * u1 - self.gravity,
                self.length / self.inertia * u2,
            ]
        )

        return np.hstack([qdot, qddot[:, 0]])


# from Tedrake's lecture notes
class LinearSatellite(GeneralCTModel):
    def __init__(self):
        super().__init__()
        # parameters based on [Bouadi, Bouchoucha, Tadjine, 2007]
        self.mass = 1.0  # mass of quadrotor
        self.gravity = 9.81  # gravity
        # data taken from
        # https://github.com/MIT-REALM/neural_clbf/
        MU = 3.986e14
        a = 42164e3
        self.n = MU / (a ** 3)

    def f_torch(self, v, u):

        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
        # with respect to the original paper, we define
        # w1 = u1+u2
        # w2 = u1-u2
        q = v[:, :3]
        qdot = v[:, 3:]
        qddot = torch.vstack(
            [
                3.0 * self.n ** 2 * q[:, 0]
                - 2.0 * self.n * qdot[:, 1]
                + u1 / self.mass,
                -2.0 * self.n * qdot[:, 0] + u2 / self.mass,
                -self.n ** 2 * q[:, 2] + u3 / self.mass,
            ]
        ).T

        return torch.hstack([qdot, qddot])

    def f_smt(self, v, u):

        u1, u2, u3 = u[0], u[1], u[2]
        # with respect to the original paper, we define
        # w1 = u1+u2
        # w2 = u1-u2
        q = v[:3]
        qdot = np.array(v[3:])
        qddot = np.array(
            [
                3.0 * self.n ** 2 * q[0] - 2.0 * self.n * qdot[1] + u1 / self.mass,
                -2.0 * self.n * qdot[0] + u2 / self.mass,
                -self.n ** 2 * q[2] + u3 / self.mass,
            ]
        )

        return np.hstack([qdot, qddot[:, 0]])


class CtrlObstacleAvoidance(GeneralCTModel):
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
        u1 = u[0][0]
        velo = 1
        sin = self.fncs["sin"]
        cos = self.fncs["cos"]
        return [
            velo * sin(phi),
            velo * cos(phi),
            -sin(phi) + u1,
        ]
