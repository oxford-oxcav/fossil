# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# definition of various activation fcns
# pylint: disable=no-member

import logging

import torch
import numpy as np
import z3
from cvc5 import pythonic as cvpy
import sympy as sp

try:
    import dreal as dr
except Exception as e:
    logging.exception("Exception while importing dReal")


from fossil import consts
from fossil.utils import contains_object


def get_symbolic_functions(v):
    if contains_object(v, dr.Variable) or contains_object(v, dr.Expression):
        return consts.DREAL_FNCS
    elif contains_object(v, z3.ArithRef):
        return consts.Z3_FNCS
    elif contains_object(v, cvpy.ArithRef):
        return consts.CVC5_FNCS
    elif contains_object(v, sp.Expr):
        return consts.SP_FNCS
    else:
        raise NotImplementedError(f"Symbolic functions for {type(v)} not implemented")


class Activation(torch.nn.Module):
    TYPE = None

    def __init__(self):
        super(Activation, self).__init__()
        self.name = self.TYPE.name

    def forward(self, x):
        raise NotImplementedError

    def forward_symbolic(self, x):
        raise NotImplementedError

    def derivative_symbolic(self, x):
        raise NotImplementedError

    def backward_symbolic(self, x):
        """Symbolic backward pass for activation function"""
        return self.derivative_symbolic(x)


class Identity(Activation):
    TYPE = consts.ActivationType.IDENTITY

    def forward(self, x):
        return x

    def forward_symbolic(self, x):
        return x

    def derivative_symbolic(self, x):
        return np.ones((x.shape))


class ReLU(Activation):
    TYPE = consts.ActivationType.RELU

    def forward(self, x):
        return torch.nn.ReLU()(x)

    def forward_symbolic(self, x):
        # Won't work with sympy
        fncs = get_symbolic_functions(x)
        ReLU = fncs["ReLU"]
        simplify = fncs["simplify"]
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = simplify(ReLU(x[idx, 0]))
        return y

    def derivative_symbolic(self, x):
        fncs = get_symbolic_functions(x)
        If = fncs["If"]
        simplify = fncs["simplify"]
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = simplify(If(x[idx, 0] > 0, 1, 0))
        return y


class Square(Activation):
    TYPE = consts.ActivationType.SQUARE

    def forward(self, x):
        return torch.pow(x, 2)

    def forward_symbolic(self, x):
        return x**2

    def derivative_symbolic(self, x):
        return 2 * x


class Poly2(Activation):
    TYPE = consts.ActivationType.POLY_2

    def forward(self, x):
        h = int(x.shape[1] / 2)
        x1, x2 = x[:, :h], x[:, h:]
        return torch.cat([x1, torch.pow(x2, 2)], dim=1)

    def forward_symbolic(self, x):
        h = int(x.shape[0] / 2)
        x1, x2 = x[:h], x[h:]
        return np.vstack([x1, np.power(x2, 2)])

    def derivative_symbolic(self, x):
        h = int(x.shape[0] / 2)
        x1, x2 = x[:h], x[h:]
        return np.vstack([np.ones(x1.shape), 2 * x2])


class Tanh(Activation):
    TYPE = consts.ActivationType.TANH

    def forward(self, x):
        return torch.tanh(x)

    def forward_symbolic(self, x):
        fncs = get_symbolic_functions(x)
        tanh = fncs["tanh"]
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = tanh(y[idx, 0])
        return y

    def derivative_symbolic(self, x):
        fncs = get_symbolic_functions(x)
        tanh = fncs["tanh"]
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = 1 - tanh(y[idx, 0]) ** 2
        return y


class TanhSquared(Activation):
    TYPE = consts.ActivationType.TANH_SQUARE

    def forward(self, x):
        return torch.pow(torch.tanh(x), 2)

    def forward_symbolic(self, x):
        fncs = get_symbolic_functions(x)
        tanh = fncs["tanh"]
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = tanh(y[idx, 0]) ** 2
        return y

    def derivative_symbolic(self, x):
        fncs = get_symbolic_functions(x)
        tanh = fncs["tanh"]
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = 2 * tanh(y[idx, 0]) * (1 - tanh(y[idx, 0]) ** 2)
        return y


class Sigmoid(Activation):
    TYPE = consts.ActivationType.SIGMOID

    def forward(self, x):
        return torch.sigmoid(x)

    def forward_symbolic(self, x):
        fncs = get_symbolic_functions(x)
        exp = fncs["exp"]
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = 1 / (1 + exp(-y[idx, 0]))
        return y

    def derivative_symbolic(self, x):
        fncs = get_symbolic_functions(x)
        exp = fncs["exp"]
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = exp(-y[idx, 0]) / ((1 + exp(-y[idx, 0])) ** 2)
        return y


class Softplus(Activation):
    TYPE = consts.ActivationType.SOFTPLUS

    def forward(self, x):
        return torch.nn.functional.softplus(x)

    def forward_symbolic(self, x):
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = dr.log(1 + dr.exp(y[idx, 0]))
        return y

    def derivative_symbolic(self, x):
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = 1 / (1 + dr.exp(-y[idx, 0]))
        return y


class ShiftedSoftplus(Activation):
    TYPE = consts.ActivationType.SHIFTED_SOFTPLUS

    def forward(self, x):
        return torch.nn.functional.softplus(x) - torch.log(torch.tensor(2.0))

    def forward_symbolic(self, x):
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = dr.log(1 + dr.exp(y[idx, 0])) - dr.log(2)
        return y

    def derivative_symbolic(self, x):
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = 1 / (1 + dr.exp(-y[idx, 0]))
        return y


class ShiftedSoftplusSquare(Activation):
    TYPE = consts.ActivationType.SHIFTED_SOFTPLUS_SQUARE

    def forward(self, x):
        return (torch.nn.functional.softplus(x) - torch.log(torch.tensor(2.0))) ** 2

    def forward_symbolic(self, x):
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = (dr.log(1 + dr.exp(y[idx, 0])) - dr.log(2)) ** 2
        return y

    def derivative_symbolic(self, x):
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = (
                2
                * dr.exp(y[idx, 0])
                * dr.log(0.5 * (1 + dr.exp(y[idx, 0])))
                / (1 + dr.exp(y[idx, 0]))
            )
        return y


class Cosh(Activation):
    TYPE = consts.ActivationType.COSH

    def forward(self, x):
        return torch.cosh(x) - 1

    def forward_symbolic(self, x):
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = dr.cosh(y[idx, 0]) - 1
        return y

    def derivative_symbolic(self, x):
        y = x.copy()
        for idx in range(len(y)):
            y[idx, 0] = dr.sinh(y[idx, 0])
        return y


class PolyN(Activation):
    def __init__(self, n, TYPE):
        self.TYPE = TYPE
        self.n = n
        super(PolyN, self).__init__()

    def forward(self, x):
        h = int(x.shape[1] / self.n)
        x_parts = [x[:, i * h : (i + 1) * h] for i in range(self.n - 1)]
        x_final = x[:, (self.n - 1) * h :]
        x_parts.append(x_final)
        return torch.cat([x_parts[i] ** (i + 1) for i in range(self.n)], dim=1)

    def forward_symbolic(self, x):
        h = int(x.shape[0] / self.n)
        x_parts = [x[i * h : (i + 1) * h] for i in range(self.n - 1)]
        x_final = x[(self.n - 1) * h :]
        x_parts.append(x_final)
        return np.vstack([x_parts[i] ** (i + 1) for i in range(self.n)])

    def derivative_symbolic(self, x):
        h = int(x.shape[0] / self.n)
        x_parts = [x[i * h : (i + 1) * h] for i in range(self.n - 1)]
        x_final = x[(self.n - 1) * h :]
        x_parts.append(x_final)
        return np.vstack([(i + 1) * x_parts[i] ** i for i in range(self.n)])


class EvenPolyN(Activation):
    def __init__(self, n, TYPE):
        self.TYPE = TYPE
        self.n = n
        super(EvenPolyN, self).__init__()

    def forward(self, x):
        h = int(x.shape[1] / self.n)
        x_parts = [x[:, i * h : (i + 1) * h] for i in range(self.n - 1)]
        x_final = x[:, (self.n - 1) * h :]
        x_parts.append(x_final)
        return torch.cat([x_parts[i] ** (2 * i) for i in range(self.n)], dim=1)

    def forward_symbolic(self, x):
        h = int(x.shape[0] / self.n)
        x_parts = [x[i * h : (i + 1) * h] for i in range(self.n - 1)]
        x_final = x[(self.n - 1) * h :]
        x_parts.append(x_final)
        return np.vstack([x_parts[i] ** (2 * i) for i in range(self.n)])

    def derivative_symbolic(self, x):
        h = int(x.shape[0] / self.n)
        x_parts = [x[i * h : (i + 1) * h] for i in range(self.n - 1)]
        x_final = x[(self.n - 1) * h :]
        x_parts.append(x_final)
        return np.vstack([(2 * i) * x_parts[i] ** (2 * i - 1) for i in range(self.n)])


def activation_fcn(select: consts.ActivationType):
    """
    :param select: enum selects the type of activation
    :return: calls the activation fcn and returns the layer after activation
    """
    if select == consts.ActivationType.IDENTITY:
        return Identity()
    elif select == consts.ActivationType.RELU:
        return ReLU()
    elif select == consts.ActivationType.LINEAR:
        return Identity()
    elif select == consts.ActivationType.SQUARE:
        return Square()
    elif select == consts.ActivationType.POLY_2:
        return Poly2()
    elif select == consts.ActivationType.RELU_SQUARE:
        return ReLU()
    elif select == consts.ActivationType.REQU:
        return ReLU()
    elif select == consts.ActivationType.TANH:
        return Tanh()
    elif select == consts.ActivationType.TANH_SQUARE:
        return TanhSquared()
    elif select == consts.ActivationType.SIGMOID:
        return Sigmoid()
    elif select == consts.ActivationType.SOFTPLUS:
        return Softplus()
    elif select == consts.ActivationType.SHIFTED_SOFTPLUS:
        return ShiftedSoftplus()
    elif select == consts.ActivationType.SHIFTED_SOFTPLUS_SQUARE:
        return ShiftedSoftplusSquare()
    elif select == consts.ActivationType.COSH:
        return Cosh()
    elif select == consts.ActivationType.POLY_3:
        return PolyN(3, select)
    elif select == consts.ActivationType.POLY_4:
        return PolyN(4, select)
    elif select == consts.ActivationType.POLY_5:
        return PolyN(5, select)
    elif select == consts.ActivationType.POLY_6:
        return PolyN(6, select)
    elif select == consts.ActivationType.POLY_7:
        return PolyN(7, select)
    elif select == consts.ActivationType.POLY_8:
        return PolyN(8, select)
    elif select == consts.ActivationType.EVEN_POLY_4:
        return EvenPolyN(4, select)
    elif select == consts.ActivationType.EVEN_POLY_6:
        return EvenPolyN(6, select)
    elif select == consts.ActivationType.EVEN_POLY_8:
        return EvenPolyN(8, select)
    elif select == consts.ActivationType.EVEN_POLY_10:
        return EvenPolyN(10, select)


# Activation function
