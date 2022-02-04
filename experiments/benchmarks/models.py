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
    # Really this should be a separate class but its just to demonstrate the idea
    def f_torch(self, v):
        x0, x1 = v[:, 0], v[:, 1]

        _then = -x1 - 0.5 * x0 ** 3
        _else = -x1 - x0 ** 2 - 0.25 * x1 ** 3
        _cond = x1 >= 0

        return torch.stack([-x0, torch.where(_cond, _then, _else)]).T

    def f_smt(self, v):
        _If = self.fncs["If"]
        x0, x1 = v
        _then = -x1 - 0.5 * x0 ** 3
        _else = -x1 - x0 ** 2 - 0.25 * x1 ** 3
        _cond = x1 >= 0
        return [-x0, _If(_cond, _then, _else)]


class Barr1(CTModel):
    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([y + 2 * x * y, -x - y ** 2 + 2 * x ** 2]).T

    def f_smt(self, v):
        x, y = v
        return [y + 2 * x * y, -x - y ** 2 + 2 * x ** 2]
