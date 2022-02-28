from typing import Any

import torch
import z3
import dreal
import numpy as np

from src.shared.utils import contains_object


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


class NonPoly0(CTModel):
    # Possibly add init with self.name attr, and maybe merge z3 & dreal funcs using dicts
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-x + x * y, -y]).T

    def f_smt(self, v):
        x, y = v
        return [-x + x * y, -y]


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
