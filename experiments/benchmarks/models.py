import sys
import inspect
from typing import Any

import numpy as np
import torch

from matplotlib import pyplot as plt

from fossil import control


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


class Linear0(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [-x - y, x]

    def f_smt(self, v):
        x, y = v
        return [-x - y, x]


class NonPoly0(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [-x + x * y, -y]

    def f_smt(self, v):
        x, y = v
        return [-x + x * y, -y]


class NonPoly1(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [-x + 2 * x**2 * y, -y]

    def f_smt(self, v):
        x, y = v
        return [-x + 2 * x**2 * y, -y]


class NonPoly2(control.CTModel):
    n_vars = 3

    def f_torch(self, v):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        return [-x, -2 * y + 0.1 * x * y**2 + z, -z - 1.5 * y]

    def f_smt(self, v):
        x, y, z = v
        return [-x, -2 * y + 0.1 * x * y**2 + z, -z - 1.5 * y]


class NonPoly3(control.CTModel):
    n_vars = 3

    def f_torch(self, v):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        return [-3 * x - 0.1 * x * y**3, -y + z, -z]

    def f_smt(self, v):
        x, y, z = v
        return [-3 * x - 0.1 * x * y**3, -y + z, -z]


class Benchmark0(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [-x, -y]

    def f_smt(self, v):
        x, y = v
        return [-x, -y]


class Benchmark1(control.ControllableCTModel):
    n_u = 2
    n_vars = 2

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2 = u[:, 0], u[:, 1]
        return [x + y + u1, -y - x + u2]

    def f_smt(self, v, u):
        x, y = v
        u1, u2 = u
        return [x + y + u1, -y - x + u2]


class Benchmark2(control.ControllableCTModel):
    n_vars = 2
    n_u = 3

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
        return [x + y + u1 - u2, y + 2.0 * x + u3]

    def f_smt(self, v, u):
        x, y = v
        u1, u2, u3 = u
        return [x + y + u1 - u2, y + 2.0 * x + u3]


class BenchmarkDT1(control.ControllableCTModel):
    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2 = u[:, 0], u[:, 1]
        return [2.0 * x + u1, 2.0 * y + u2]

    def f_smt(self, v, u):
        x, y = v
        u1, u2 = u
        return [2.0 * x + u1, 2.0 * y + u2]


# POLY benchmarks
# this series comes from
# https://www.cs.colorado.edu/~srirams/papers/nolcos13.pdf
# srirams paper from 2013 (old-ish) but plenty of lyap fcns


class Poly1(control.CTModel):
    n_vars = 3

    def f_torch(self, v):
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        return [-(x**3) - x * z**2, -y - x**2 * y, -z + 3 * x**2 * z - (3 * z)]

    def f_smt(self, v):
        x, y, z = v
        return [-(x**3) - x * z**2, -y - x**2 * y, -z + 3 * x**2 * z - (3 * z)]


class Poly2(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [-(x**3) + y, -x - y]

    def f_smt(self, v):
        x, y = v
        return [-(x**3) + y, -x - y]


class Poly3(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [-(x**3) - y**2, x * y - y**3]

    def f_smt(self, v):
        x, y = v
        return [-(x**3) - y**2, x * y - y**3]


class Poly4(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [-x - 1.5 * x**2 * y**3, -(y**3) + 0.5 * x**3 * y**2]

    def f_smt(self, v):
        x, y = v
        return [-x - 1.5 * x**2 * y**3, -(y**3) + 0.5 * x**3 * y**2]


class Sriram4D(control.CTModel):
    n_vars = 4

    def f_torch(self, v):
        x1, x2, x3, x4 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        return [
            -x1 + x2**3 - 3 * x3 * x4,
            -x1 - x2**3,
            x1 * x4 - x3,
            x1 * x3 - x4**3,
        ]

    def f_smt(self, v):
        x1, x2, x3, x4 = v
        return [
            -x1 + x2**3 - 3 * x3 * x4,
            -x1 - x2**3,
            x1 * x4 - x3,
            x1 * x3 - x4**3,
        ]


class LinearDiscrete(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [0.5 * x - 0.5 * y, 0.5 * x]

    def f_smt(self, v):
        x, y = v
        return [0.5 * x - 0.5 * y, 0.5 * x]


class DoubleLinearDiscrete(control.CTModel):
    n_vars = 4

    def f_torch(self, v):
        x1, x2, x3, x4 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        return [0.5 * x1 - 0.5 * x2, 0.5 * x1, 0.5 * x3 - 0.5 * x4, 0.5 * x3]

    def f_smt(self, v):
        x1, x2, x3, x4 = v
        return [0.5 * x1 - 0.5 * x2, 0.5 * x1, 0.5 * x3 - 0.5 * x4, 0.5 * x3]


class LinearDiscreteNVars(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [0.5 * v[i] for i in range(len(v))]

    def f_smt(self, v):
        x, y = v
        return [0.5 * v[i] for i in range(len(v))]


class NonLinearDiscrete(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [0.5 * x - 0.000001 * y**2, 0.5 * x * y]

    def f_smt(self, v):
        x, y = v
        return [0.5 * x - 0.000001 * y**2, 0.5 * x * y]


# class TwoDHybrid(control.CTModel):
#     n_vars = 2

#     def f_torch(self, v):
#         x0, x1 = v[:, 0], v[:, 1]
#         _condition = x1 >= 0
#         _negated_cond = x1 < 0
#         _then = -x1 - 0.5 * x0**3
#         _else = -x1 - x0**2 - 0.25 * x1**3
#         # _condition and _negated _condition are tensors of bool, act like 0 and 1
#         x1dot = _condition * _then + _negated_cond * _else

#         return [-x0, x1dot]

#     def f_smt(self, v):
#         _If = self.fncs["If"]
#         x0, x1 = v
#         _then = -x1 - 0.5 * x0**3
#         _else = -x1 - x0**2 - 0.25 * x1**3
#         _cond = x1 >= 0
#         return [-x0, _If(_cond, _then, _else)]

# class TwoD_Hybrid(control.CTModel):
#     n_vars = 2

#     def f_torch(self, v):
#         x0, x1 = v[:, 0], v[:, 1]
#         _then = -x0 - 0.5 * x0**3
#         _else = x0 - 0.25 * x1**2
#         _cond = x0 >= 0
#         return [x1, torch.where(_cond, _then, _else)]

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


class Barr1(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [y + 2 * x * y, -x - y**2 + 2 * x**2]

    def f_smt(self, v):
        x, y = v
        return [y + 2 * x * y, -x - y**2 + 2 * x**2]


class Barr2(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [torch.exp(-x) + y - 1, -(torch.sin(x) ** 2)]

    def f_smt(self, v):
        sin = self.fncs["sin"]
        exp = self.fncs["exp"]
        x, y = v
        return [exp(-x) + y - 1, -((sin(x)) ** 2)]


class Barr3(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [y, -x - y + 1 / 3 * x**3]

    def f_smt(self, v):
        x, y = v
        return [y, -x - y + 1 / 3 * x**3]


class ObstacleAvoidance(control.CTModel):
    n_vars = 3

    def f_torch(self, v):
        x, y, phi = v[:, 0], v[:, 1], v[:, 2]
        velo = 1
        return [
            velo * torch.sin(phi),
            velo * torch.cos(phi),
            -torch.sin(phi)
            + 3 * (x * torch.sin(phi) + y * torch.cos(phi)) / (0.5 + x**2 + y**2),
        ]

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


class HighOrd4(control.CTModel):
    n_vars = 4

    def f_torch(self, v):
        x0, x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        return [x1, x2, x3, -3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0]

    def f_smt(self, v):
        x0, x1, x2, x3 = v
        return [x1, x2, x3, -3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0]


class HighOrd6(control.CTModel):
    n_vars = 6

    def f_torch(self, v):
        x0, x1, x2, x3, x4, x5 = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5]
        return [
            x1,
            x2,
            x3,
            x4,
            x5,
            -800 * x5 - 2273 * x4 - 3980 * x3 - 4180 * x2 - 2400 * x1 - 576 * x0,
        ]

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


class HighOrd8(control.CTModel):
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


class UnstableLinear(control.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [-2 * x - y, 0.6 * y]

    def f_smt(self, v):
        x, y = v
        return [-2 * x - y, 0.6 * y]


class Car(control.CTModel):
    n_vars = 3

    def f_torch(self, v):
        x, y, omega = v[:, 0], v[:, 1], v[:, 2]
        return [torch.cos(omega), torch.sin(omega), omega]

    def f_smt(self, v):
        x, y, omega = v
        sin = self.fncs["sin"]
        cos = self.fncs["cos"]
        return [cos(omega), sin(omega), omega]


class InvertedPendulum(control.ControllableCTModel):
    n_vars = 2
    n_u = 2

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2 = u[:, 0], u[:, 1]

        G = 9.81  # gravity
        L = 0.5  # length of the pole
        m = 0.15  # ball mass
        b = 0.1  # friction

        return [y + u1, u2 + (m * G * L * torch.sin(x) - b * y) / (m * L**2)]

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


class InvertedPendulumLQR(control.CTModel):
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

        return [y + u1, u2 + (m * G * L * torch.sin(x) - b * y) / (m * L**2)]

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


class LorenzSystem(control.ControllableCTModel):
    n_vars = 3
    n_u = 3

    def f_torch(self, v, u):
        x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2]
        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]

        sigma = 10.0  # related to Prandtl number,
        r = 28.0  # related to Rayleigh number,
        b = 8.0 / 3.0  # geometric factor (Weisstein, 2002).

        return [
            -sigma * (x1 - x2) + u1,
            r * x1 - x2 - x1 * x3 + u2,
            x1 * x2 - b * x3 + u3,
        ]

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


class CtrlCar(control.ControllableCTModel):
    n_vars = 3

    def f_torch(self, v, u):
        x, y, omega = v[:, 0], v[:, 1], v[:, 2]
        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
        return [torch.cos(omega) + u1, torch.sin(omega) + u2, omega + u3]

    def f_smt(self, v, u):
        x, y, omega = v
        u1, u2, u3 = u
        sin = self.fncs["sin"]
        cos = self.fncs["cos"]
        return [cos(omega) + u1, sin(omega) + u2, omega + u3]


# from Tedrake's lecture notes
class Quadrotor2d(control.ControllableCTModel):
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
class LinearSatellite(control.ControllableCTModel):
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


class CtrlObstacleAvoidance(control.ControllableCTModel):
    n_vars = 3
    n_u = 1

    def f_torch(self, v, u):
        x, y, phi = v[:, 0], v[:, 1], v[:, 2]
        u1 = u[:, 0]
        velo = 1
        return [
            velo * torch.sin(phi),
            velo * torch.cos(phi),
            -torch.sin(phi) + u1,
        ]

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


class Identity(control.ControllableCTModel):
    n_vars = 2
    n_u = 2

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2 = u[:, 0], u[:, 1]
        return [x + u1, y + u2]

    def f_smt(self, v, u):
        x, y = v
        u1, u2 = u
        return [x + u1, y + u2]


class DTAhmadi(control.ControllableCTModel):
    # from Non-monotonic Lyapunov Functions
    # for Stability of Discrete Time Nonlinear and Switched Systems
    # Amir Ali Ahmadi and Pablo A. Parrilo, CDC 2008.
    n_vars = 2
    n_u = 2

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1, u2 = u[:, 0], u[:, 1]
        return [0.3 * x + u1, -x + 0.5 * y + 7.0 / 18.0 * x**2 + u2]

    def f_smt(self, v, u):
        x, y = v
        u1, u2 = u
        return [0.3 * x + u1, -x + 0.5 * y + 7.0 / 18.0 * x**2 + u2]


### Benchmarks taken from RSWS work of Verdier, Mazo


class Linear1(control.ControllableCTModel):
    n_vars = 2
    n_u = 1

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1 = u[:, 0]
        return [y, -x + u1]

    def f_smt(self, v, u):
        x, y = v
        (u1,) = u
        return [y, -x + u1]


class Linear1LQR(control.CTModel):
    """Linear1 with LQR controller"""

    n_vars = 2
    K = [0.414, 1.352]

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        u1 = -(self.K[0] * x + self.K[1] * y)
        return [y, -x + u1]

    def f_smt(self, v):
        x, y = v
        u1 = -(self.K[0] * x + self.K[1] * y)
        return [y, -x + u1]


class SecondOrder(control.ControllableCTModel):
    n_vars = 2
    n_u = 1

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1 = u[:, 0]
        return [y - x**3, u1]

    def f_smt(self, v, u):
        x, y = v
        (u1,) = u
        return [y - x**3, u1]


class SecondOrderLQR(control.CTModel):
    n_vars = 2
    K = [1.0, 1.73]

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        u1 = -(self.K[0] * x + self.K[1] * y)
        return [y - x**3, u1]

    def f_smt(self, v):
        x, y = v
        u1 = -(self.K[0] * x + self.K[1] * y)
        return [y - x**3, u1]


class ThirdOrder(control.ControllableCTModel):
    n_vars = 3
    n_u = 1

    def f_torch(self, v, u):
        x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2]
        u1 = u[:, 0]
        return [-10 * x1 + 10 * x2 + u1, 28 * x1 - x2 - x1 * x3, x1 * x2 - 8 / 3 * x3]

    def f_smt(self, v, u):
        x1, x2, x3 = v
        (u1,) = u
        return [-10 * x1 + 10 * x2 + u1, 28 * x1 - x2 - x1 * x3, x1 * x2 - 8 / 3 * x3]


class ThirdOrderLQR(control.CTModel):
    n_vars = 3
    K = [23.71, 18.49, 0.0]

    def f_torch(self, v):
        x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2]
        u1 = -(self.K[0] * x1 + self.K[1] * x2 + self.K[2] * x3)
        return [-10 * x1 + 10 * x2 + u1, 28 * x1 - x2 - x1 * x3, x1 * x2 - 8 / 3 * x3]

    def f_smt(self, v):
        x1, x2, x3 = v
        u1 = -(self.K[0] * x1 + self.K[1] * x2 + self.K[2] * x3)
        return [-10 * x1 + 10 * x2 + u1, 28 * x1 - x2 - x1 * x3, x1 * x2 - 8 / 3 * x3]


class LoktaVolterra(control.CTModel):
    n_vars = 2

    def __init__(self) -> None:
        super().__init__()

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [0.6 * x - x * y, -0.6 * y + x * y]

    def f_smt(self, v):
        x, y = v
        return [0.6 * x - x * y, -0.6 * y + x * y]


class VanDerPol(control.CTModel):
    n_vars = 2

    def __init__(self) -> None:
        super().__init__()

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [y, -0.5 * (1 - x**2) * y - x]

    def f_smt(self, v):
        x, y = v
        return [y, -0.5 * (1 - x**2) * y - x]


class SheModel(control.CTModel):
    n_vars = 6

    def f_torch(self, v):
        x1, x2, x3, x4, x5, x6 = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5]
        return [
            x2 * x4 - x1**3,
            -3 * x1 * x4 - x2**3,
            -x3 - 3 * x1 * x4**3,
            -x4 + x1 * x3,
            -x5 + x6**3,
            -x5 - x6 + x3**4,
        ]

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


class PapaPrajna6(control.CTModel):
    n_vars = 6

    def f_torch(self, v):
        x1, x2, x3, x4, x5, x6 = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5]
        return [
            -(x1**3) + 4 * x2**3 - 6 * x3 * x4,
            -x1 - x2 + x5**3,
            x1 * x4 - x3 + x4 * x6,
            x1 * x3 + x3 * x6 - x4**3,
            -2 * x2**3 - x5 + x6,
            -3 * x3 * x4 - x5**3 - x6,
        ]

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


def read_model(model_string: str) -> control.CTModel:
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
            if isinstance(m, control.CTModel):
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
