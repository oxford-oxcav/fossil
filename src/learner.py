# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Literal
import warnings

import numpy as np
import torch
import torch.nn as nn

from src.shared.activations import ActivationType, activation, activation_der
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.shared.component import Component
from src.shared.consts import LearningFactors, TimeDomain
from src.shared.utils import Timer, timer
from src.shared.activations_symbolic import activation_sym
from src.shared.control import GeneralController

T = Timer()


class QuadraticFactor(nn.Module):
    def forward(self, x):
        return torch.pow(x, 2).sum(dim=1)

    def derivative(self, x):
        return 2 * x


class Learner(Component):
    def __init__(self):
        super().__init__()

    def get(self, **kw):
        return self.learn(**kw)

    def learn(self, *args, **kwargs):
        return NotImplemented("Not implemented in " + self.__class__.__name__)


class LearnerCT(nn.Module, Learner):
    def __init__(
        self,
        input_size,
        learn_method,
        *args,
        bias=True,
        activate=ActivationType.SQUARE,
        equilibria=0,
        llo=False,
        **kw
    ):
        super(LearnerCT, self).__init__()

        self.filter = True
        self.input_size = input_size
        n_prev = self.input_size
        self.eq = equilibria
        self._diagonalise = False
        self.acts = activate
        self._is_there_bias = bias
        self.verbose = kw.get(CegisConfig.VERBOSE.k, CegisConfig.VERBOSE.v)
        ZaZ = kw.get(CegisConfig.FACTORS.k, CegisConfig.FACTORS.v)
        self.factor = QuadraticFactor() if ZaZ == LearningFactors.QUADRATIC else None
        self.layers = []
        self.closest_unsat = None
        k = 1

        for n_hid in args:
            layer = nn.Linear(n_prev, n_hid, bias=bias)
            self.register_parameter("W" + str(k), layer.weight)
            if bias:
                self.register_parameter("b" + str(k), layer.bias)
            self.layers.append(layer)
            n_prev = n_hid
            k = k + 1

            # last layer
        layer = nn.Linear(n_prev, 1, bias=False)
        # last layer of ones
        if llo:
            layer.weight = torch.nn.Parameter(torch.ones(layer.weight.shape))
            self.layers.append(layer)
        else:  # free output layer
            self.register_parameter("W" + str(k), layer.weight)
            self.layers.append(layer)
        self.learn_method = learn_method

    @staticmethod
    def learner_fncts():
        return {
            "sin": torch.sin,
            "cos": torch.cos,
            "exp": torch.exp,
            "If": lambda cond, _then, _else: _then if cond.item() else _else,
        }

    def forward(self, S, Sdot):
        assert len(S) == len(Sdot)

        nn, grad_nn = self.forward_tensors(S)
        # circle = x0*x0 + ... + xN*xN
        circle = torch.pow(S, 2).sum(dim=1)

        F, derivative_F = self.compute_factors(S)
        V = nn * F
        # define F(x) := ||x||^2
        # V = NN(x) * F(x)
        # gradV = NN(x) * dF(x)/dx  + der(NN) * F(x)
        # gradV = torch.stack([nn, nn]).T * derivative_e + grad_nn * torch.stack([E, E]).T
        if derivative_F != 0:
            gradV = (
                nn.expand_as(grad_nn.T).T * derivative_F.expand_as(grad_nn)
                + grad_nn * F.expand_as(grad_nn.T).T
            )
        else:
            gradV = grad_nn
        # Vdot = gradV * f(x)
        Vdot = torch.sum(torch.mul(gradV, Sdot), dim=1)

        return V, Vdot, circle

    # generalisation of forward with tensors
    def forward_tensors(self, x):
        """
        :param x: tensor of data points
        :return:
                V: tensor, evaluation of x in net
                jacobian: tensor, evaluation of grad_net
        """
        y = x
        jacobian = torch.diag_embed(torch.ones(x.shape[0], self.input_size))

        for idx, layer in enumerate(self.layers[:-1]):
            z = layer(y)
            y = activation(self.acts[idx], z)

            jacobian = torch.matmul(layer.weight, jacobian)
            jacobian = torch.matmul(
                torch.diag_embed(activation_der(self.acts[idx], z)), jacobian
            )

        numerical_v = torch.matmul(y, self.layers[-1].weight.T)
        jacobian = torch.matmul(self.layers[-1].weight, jacobian)

        return numerical_v[:, 0], jacobian[:, 0, :]

    def compute_factors(self, S):
        if self.factor:
            return self.factor(S), self.factor.derivative(S)
        else:
            return 1, 0

    def get(self, **kw):
        return self.learn(
            kw[CegisStateKeys.optimizer], kw[CegisStateKeys.S], kw[CegisStateKeys.S_dot]
        )

    # backprop algo
    @timer(T)
    def learn(self, optimizer, S, Sdot):
        return self.learn_method(self, optimizer, S, Sdot)

    def diagonalisation(self):
        # makes the weight matrices diagonal. works iff intermediate layers are square matrices
        with torch.no_grad():
            for layer in self.layers[:-1]:
                layer.weight.data = torch.diag(torch.diag(layer.weight))

    def find_closest_unsat(self, S, Sdot):
        min_dist = float("inf")
        V, Vdot, _ = self.forward(S[0], Sdot[0])
        for iii in range(S[0].shape[0]):
            v = V[iii]
            vdot = Vdot[iii]
            if V[iii].item() < 0 or Vdot[iii].item() > 0:
                dist = S[0][iii].norm()
                if dist < min_dist:
                    min_dist = dist
        self.closest_unsat = min_dist

    @staticmethod
    def is_positive_definite(activations: list, layers: list):
        """Checks if the net is positive definite, assuming W is also positive definite;

        Positive definiteness is defined as the following:
        N(x) > 0 for all x in R^n

        Args:
            activations (list): list of activation functions used in the net
            layers (list): list of layers in the net

        Returns:
            bool: True is net is positive definite, else False
        """
        pd_acts = [
            ActivationType.RELU,
            ActivationType.SQUARE,
            ActivationType.COSH,
            ActivationType.SIGMOID,
            ActivationType.SQUARE_DEC,
        ]
        return (activations[-1] in pd_acts) and (layers[-1].bias is None)

    @staticmethod
    def order_of_magnitude(number):
        if number.item() != 0:
            return np.ceil(np.log10(number))
        else:
            return 1.0

    @staticmethod
    def get_timer():
        return T


class LearnerDT(nn.Module, Learner):
    def __init__(
        self,
        input_size,
        learn_method,
        *args,
        bias=True,
        activate=[ActivationType.LIN_SQUARE],
        equilibria=0,
        llo=False,
        **kw
    ):
        super(LearnerDT, self).__init__()

        self.input_size = input_size
        n_prev = input_size
        self.eq = equilibria
        self._diagonalise = False
        self.acts = activate
        self._is_there_bias = bias
        self.verbose = kw.get(CegisConfig.VERBOSE.k, CegisConfig.VERBOSE.v)
        self.factors = kw.get(CegisConfig.FACTORS.k, CegisConfig.FACTORS.v)
        self.layers = []
        self.closest_unsat = None
        k = 1
        for n_hid in args:
            layer = nn.Linear(n_prev, n_hid, bias=bias)
            self.register_parameter("W" + str(k), layer.weight)
            if bias:
                self.register_parameter("b" + str(k), layer.bias)
            self.layers.append(layer)
            n_prev = n_hid
            k = k + 1

        # last layer
        layer = nn.Linear(n_prev, 1, bias=False)
        # last layer of ones
        if llo:
            layer.weight = torch.nn.Parameter(torch.ones(layer.weight.shape))
            self.layers.append(layer)
        else:  # free output layer
            self.register_parameter("W" + str(k), layer.weight)
            self.layers.append(layer)
        self.learn_method = learn_method

    @staticmethod
    def learner_fncts():
        return {
            "sin": torch.sin,
            "cos": torch.cos,
            "exp": torch.exp,
            "If": lambda cond, _then, _else: _then if cond.item() else _else,
            "Pow": torch.pow,
            "Real": lambda x: x,
        }

    # generalisation of forward with tensors
    def numerical_net(self, x):
        """
        :param x: tensor of data points
        :param xdot: tensor of data points
        :return:
                V: tensor, evaluation of x in net
                Vdot: tensor, evaluation of x in derivative net
                jacobian: tensor, evaluation of grad_net
        """
        y = x

        for idx, layer in enumerate(self.layers[:-1]):
            z = layer(y)
            y = activation(self.acts[idx], z)

        last_layer = self.layers[-1]
        numerical_v = last_layer(y)

        return numerical_v

    def forward(self, S, Sdot):
        # assert (len(S) == len(Sdot))  ## This causes a warning in Marabou

        nn = self.numerical_net(S)
        nn_next = self.numerical_net(Sdot)
        # circle = x0*x0 + ... + xN*xN
        circle = torch.pow(S, 2).sum(dim=1)

        E = self.compute_factors(S, self.factors)

        # define E(x) := (x-eq_0) * ... * (x-eq_N)
        # V = NN(x) * E(x)
        V = nn
        delta_V = nn_next - V

        return V, delta_V, circle

    def compute_factors(self, S, lf):
        E = 1
        with torch.no_grad():
            if lf == LearningFactors.QUADRATIC:  # quadratic factors
                # define a tensor to store all the components of the quadratic factors
                # factors[:,:,0] stores [ x-x_eq0, y-y_eq0 ]
                # factors[:,:,1] stores [ x-x_eq1, y-y_eq1 ]
                for idx in range(self.eq.shape[0]):
                    # S - self.eq == [ x-x_eq, y-y_eq ]
                    # torch.power(S - self.eq, 2) == [ (x-x_eq)**2, (y-y_eq)**2 ]
                    # (vector_x - eq_0)**2 =  (x-x_eq)**2 + (y-y_eq)**2
                    E *= torch.sum(
                        torch.pow(S - torch.tensor(self.eq[idx, :]), 2), dim=1
                    )
            else:
                E = torch.tensor(
                    1.0
                )  # This also causes a warning in Marabou - but always constant so should be okay

        return E

    def get(self, **kw):
        return self.learn(
            kw[CegisStateKeys.optimizer],
            kw[CegisStateKeys.S],
            kw[CegisStateKeys.S_dot],
            kw[CegisStateKeys.factors],
        )

    # backprop algo
    @timer(T)
    def learn(self, optimizer, S, Sdot, factors):
        return self.learn_method(self, optimizer, S, Sdot)

    def diagonalisation(self):
        # makes the weight matrices diagonal. works iff intermediate layers are square matrices
        with torch.no_grad():
            for layer in self.layers[:-1]:
                layer.weight.data = torch.diag(torch.diag(layer.weight))

    def find_closest_unsat(self, S, Sdot, factors):
        min_dist = float("inf")
        V, Vdot, _ = self.numerical_net(S, Sdot, factors)
        for iii in range(S.shape[0]):
            v = V[iii]
            vdot = Vdot[iii]
            if V[iii].item() < 0 or Vdot[iii].item() > 0:
                dist = S[iii].norm()
                if dist < min_dist:
                    min_dist = dist
        self.closest_unsat = min_dist

    @staticmethod
    def order_of_magnitude(number):
        if number.item() != 0:
            return torch.ceil(torch.log10(number))
        else:
            return 1.0

    @staticmethod
    def get_timer():
        return T


"""
CtrlLearnerCT *should* give a Control Lyapunov function 
"""
# TODO: in devel
class CtrlLearnerCT(nn.Module, Learner):
    def __init__(
        self,
        input_size,
        learn_method,
        *args,
        bias=True,
        activate=ActivationType.SQUARE,
        equilibria=0,
        llo=False,
        **kw
    ):
        super(CtrlLearnerCT, self).__init__()

        self.input_size = input_size
        n_prev = input_size
        self.eq = equilibria
        self._diagonalise = False
        self.acts = activate
        self._is_there_bias = bias
        self.verbose = kw.get(CegisConfig.VERBOSE.k, CegisConfig.VERBOSE.v)
        self.factors = kw.get(CegisConfig.FACTORS.k, CegisConfig.FACTORS.v)
        self.layers = []
        self.closest_unsat = None
        self.ctrl_layers = kw.get(CegisConfig.CTRLAYER.k, CegisConfig.CTRLAYER.v)

        k = 1
        for n_hid in args:
            layer = nn.Linear(n_prev, n_hid, bias=bias)
            self.register_parameter("W" + str(k), layer.weight)
            if bias:
                self.register_parameter("b" + str(k), layer.bias)
            self.layers.append(layer)
            n_prev = n_hid
            k = k + 1

        # last layer
        layer = nn.Linear(n_prev, 1, bias=False)
        # last layer of ones
        if llo:
            layer.weight = torch.nn.Parameter(torch.ones(layer.weight.shape))
            self.layers.append(layer)
        else:  # free output layer
            self.register_parameter("W" + str(k), layer.weight)
            self.layers.append(layer)
        self.learn_method = learn_method

        # if self.ctrl_layers is not None:
        #     self.ctrler = GeneralController(inputs=input_size, output=self.ctrl_layers[-1],
        #                                     layers=self.ctrl_layers[:-1],
        #                                     activations=[ActivationType.LINEAR]*len(self.ctrl_layers))

    @staticmethod
    def learner_fncts():
        return {
            "sin": torch.sin,
            "cos": torch.cos,
            "exp": torch.exp,
            "If": lambda cond, _then, _else: _then if cond.item() else _else,
        }

    def forward(self, S, Sdot):
        assert len(S) == len(Sdot)

        nn, grad_nn = self.forward_tensors(S)
        # circle = x0*x0 + ... + xN*xN
        circle = torch.pow(S, 2).sum(dim=1)

        E, derivative_e = self.compute_factors(S, self.factors)

        # define E(x) := (x-eq_0) * ... * (x-eq_N)
        # V = NN(x) * E(x)
        V = nn * E
        # gradV = NN(x) * dE(x)/dx  + der(NN) * E(x)
        # gradV = torch.stack([nn, nn]).T * derivative_e + grad_nn * torch.stack([E, E]).T
        gradV = (
            nn.expand_as(grad_nn.T).T * derivative_e.expand_as(grad_nn)
            + grad_nn * E.expand_as(grad_nn.T).T
        )
        # Vdot = gradV * f(x)
        Vdot = torch.sum(torch.mul(gradV, Sdot), dim=1)

        return V, Vdot, circle

    # generalisation of forward with tensors
    def forward_tensors(self, x):
        """
        :param x: tensor of data points
        :return:
                V: tensor, evaluation of x in net
                jacobian: tensor, evaluation of grad_net
        """
        y = x
        jacobian = torch.diag_embed(torch.ones(x.shape[0], self.input_size))

        for idx, layer in enumerate(self.layers[:-1]):
            z = layer(y)
            y = activation(self.acts[idx], z)

            jacobian = torch.matmul(layer.weight, jacobian)
            jacobian = torch.matmul(
                torch.diag_embed(activation_der(self.acts[idx], z)), jacobian
            )

        numerical_v = torch.matmul(y, self.layers[-1].weight.T)
        jacobian = torch.matmul(self.layers[-1].weight, jacobian)

        return numerical_v[:, 0], jacobian[:, 0, :]

    def compute_factors(self, S, lf):
        E, factors = 1, []
        with torch.no_grad():
            if lf == LearningFactors.QUADRATIC:  # quadratic factors
                # define a tensor to store all the components of the quadratic factors
                # factors[:,:,0] stores [ x-x_eq0, y-y_eq0 ]
                # factors[:,:,1] stores [ x-x_eq1, y-y_eq1 ]
                factors = torch.zeros(S.shape[0], self.input_size, self.eq.shape[0])
                for idx in range(self.eq.shape[0]):
                    # S - self.eq == [ x-x_eq, y-y_eq ]
                    # torch.power(S - self.eq, 2) == [ (x-x_eq)**2, (y-y_eq)**2 ]
                    # (vector_x - eq_0)**2 =  (x-x_eq)**2 + (y-y_eq)**2
                    factors[:, :, idx] = S - torch.tensor(self.eq[idx, :])
                    E *= torch.sum(
                        torch.pow(S - torch.tensor(self.eq[idx, :]), 2), dim=1
                    )

                # derivative = 2*(x-eq)*E/E_i
                grad_e = torch.zeros(S.shape[0], self.input_size)
                for var in range(self.input_size):
                    for idx in range(self.eq.shape[0]):
                        grad_e[:, var] += (
                            E
                            * factors[:, var, idx]
                            / torch.sum(
                                torch.pow(S - torch.tensor(self.eq[idx, :]), 2), dim=1
                            )
                        )
                derivative_e = 2 * grad_e
            else:
                E, derivative_e = torch.tensor(1.0), torch.tensor(0.0)

        return E, derivative_e

    def get(self, **kw):
        return self.learn(
            kw[CegisStateKeys.optimizer],
            kw[CegisStateKeys.S],
            kw[CegisStateKeys.S_dot],
            kw[CegisStateKeys.xdot_func],
        )

    # backprop algo
    @timer(T)
    def learn(self, optimizer, S, Sdot, xdot_func):
        return self.learn_method(self, optimizer, S, Sdot, xdot_func)

    def diagonalisation(self):
        # makes the weight matrices diagonal. works iff intermediate layers are square matrices
        with torch.no_grad():
            for layer in self.layers[:-1]:
                layer.weight.data = torch.diag(torch.diag(layer.weight))

    def find_closest_unsat(self, S, Sdot):
        min_dist = float("inf")
        V, Vdot, _ = self.forward(S[0], Sdot[0])
        for iii in range(S[0].shape[0]):
            v = V[iii]
            vdot = Vdot[iii]
            if V[iii].item() < 0 or Vdot[iii].item() > 0:
                dist = S[0][iii].norm()
                if dist < min_dist:
                    min_dist = dist
        self.closest_unsat = min_dist

    @staticmethod
    def order_of_magnitude(number):
        if number.item() != 0:
            return np.ceil(np.log10(number))
        else:
            return 1.0

    @staticmethod
    def get_timer():
        return T


class CtrlLearnerDT(nn.Module, Learner):
    def __init__(
        self,
        input_size,
        learn_method,
        *args,
        bias=True,
        activate=ActivationType.SQUARE,
        equilibria=0,
        llo=False,
        **kw
    ):
        super(CtrlLearnerDT, self).__init__()

        self.input_size = input_size
        n_prev = input_size
        self.eq = equilibria
        self._diagonalise = False
        self.acts = activate
        self._is_there_bias = bias
        self.verbose = kw.get(CegisConfig.VERBOSE.k, CegisConfig.VERBOSE.v)
        self.factors = kw.get(CegisConfig.FACTORS.k, CegisConfig.FACTORS.v)
        self.layers = []
        self.closest_unsat = None
        self.ctrl_layers = kw.get(CegisConfig.CTRLAYER.k, CegisConfig.CTRLAYER.v)

        k = 1
        for n_hid in args:
            layer = nn.Linear(n_prev, n_hid, bias=bias)
            self.register_parameter("W" + str(k), layer.weight)
            if bias:
                self.register_parameter("b" + str(k), layer.bias)
            self.layers.append(layer)
            n_prev = n_hid
            k = k + 1

        # last layer
        layer = nn.Linear(n_prev, 1, bias=False)
        # last layer of ones
        if llo:
            layer.weight = torch.nn.Parameter(torch.ones(layer.weight.shape))
            self.layers.append(layer)
        else:  # free output layer
            self.register_parameter("W" + str(k), layer.weight)
            self.layers.append(layer)
        self.learn_method = learn_method

        # if self.ctrl_layers is not None:
        #     self.ctrler = GeneralController(inputs=input_size, output=self.ctrl_layers[-1],
        #                                     layers=self.ctrl_layers[:-1],
        #                                     activations=[ActivationType.LINEAR]*len(self.ctrl_layers))

    @staticmethod
    def learner_fncts():
        return {
            "sin": torch.sin,
            "cos": torch.cos,
            "exp": torch.exp,
            "If": lambda cond, _then, _else: _then if cond.item() else _else,
        }

    def numerical_net(self, x):
        """
        :param x: tensor of data points
        :param xdot: tensor of data points
        :return:
                V: tensor, evaluation of x in net
                Vdot: tensor, evaluation of x in derivative net
                jacobian: tensor, evaluation of grad_net
        """
        y = x

        for idx, layer in enumerate(self.layers[:-1]):
            z = layer(y)
            y = activation(self.acts[idx], z)

        last_layer = self.layers[-1]
        numerical_v = last_layer(y)

        return numerical_v

    def forward(self, S, Sdot):
        assert len(S) == len(Sdot)

        nn = self.numerical_net(S)
        nn_next = self.numerical_net(Sdot)
        # circle = x0*x0 + ... + xN*xN
        circle = torch.pow(S, 2).sum(dim=1)

        E = self.compute_factors(S, self.factors)

        # define E(x) := (x-eq_0) * ... * (x-eq_N)
        # V = NN(x) * E(x)
        V = nn
        delta_V = nn_next - V

        return V, delta_V, circle

    def compute_factors(self, S, lf):
        E, factors = 1, []
        with torch.no_grad():
            if lf == LearningFactors.QUADRATIC:  # quadratic factors
                # define a tensor to store all the components of the quadratic factors
                # factors[:,:,0] stores [ x-x_eq0, y-y_eq0 ]
                # factors[:,:,1] stores [ x-x_eq1, y-y_eq1 ]
                factors = torch.zeros(S.shape[0], self.input_size, self.eq.shape[0])
                for idx in range(self.eq.shape[0]):
                    # S - self.eq == [ x-x_eq, y-y_eq ]
                    # torch.power(S - self.eq, 2) == [ (x-x_eq)**2, (y-y_eq)**2 ]
                    # (vector_x - eq_0)**2 =  (x-x_eq)**2 + (y-y_eq)**2
                    factors[:, :, idx] = S - torch.tensor(self.eq[idx, :])
                    E *= torch.sum(
                        torch.pow(S - torch.tensor(self.eq[idx, :]), 2), dim=1
                    )

                # derivative = 2*(x-eq)*E/E_i
                grad_e = torch.zeros(S.shape[0], self.input_size)
                for var in range(self.input_size):
                    for idx in range(self.eq.shape[0]):
                        grad_e[:, var] += (
                            E
                            * factors[:, var, idx]
                            / torch.sum(
                                torch.pow(S - torch.tensor(self.eq[idx, :]), 2), dim=1
                            )
                        )
                derivative_e = 2 * grad_e
            else:
                E, derivative_e = torch.tensor(1.0), torch.tensor(0.0)

        return E, derivative_e

    def get(self, **kw):
        return self.learn(
            kw[CegisStateKeys.optimizer],
            kw[CegisStateKeys.S],
            kw[CegisStateKeys.S_dot],
            kw[CegisStateKeys.xdot_func],
        )

    # backprop algo
    @timer(T)
    def learn(self, optimizer, S, Sdot, xdot_func):
        return self.learn_method(self, optimizer, S, Sdot, xdot_func)

    def diagonalisation(self):
        # makes the weight matrices diagonal. works iff intermediate layers are square matrices
        with torch.no_grad():
            for layer in self.layers[:-1]:
                layer.weight.data = torch.diag(torch.diag(layer.weight))

    def find_closest_unsat(self, S, Sdot):
        min_dist = float("inf")
        V, Vdot, _ = self.forward(S[0], Sdot[0])
        for iii in range(S[0].shape[0]):
            v = V[iii]
            vdot = Vdot[iii]
            if V[iii].item() < 0 or Vdot[iii].item() > 0:
                dist = S[0][iii].norm()
                if dist < min_dist:
                    min_dist = dist
        self.closest_unsat = min_dist

    @staticmethod
    def order_of_magnitude(number):
        if number.item() != 0:
            return np.ceil(np.log10(number))
        else:
            return 1.0

    @staticmethod
    def get_timer():
        return T


def get_learner(time_domain: Literal, ctrl: Literal) -> Learner:
    if ctrl and time_domain == TimeDomain.CONTINUOUS:
        return CtrlLearnerCT
    elif ctrl and time_domain == TimeDomain.DISCRETE:
        return CtrlLearnerDT
    elif time_domain == TimeDomain.CONTINUOUS:
        return LearnerCT
    elif time_domain == TimeDomain.DISCRETE:
        return LearnerDT
    else:
        raise ValueError("Learner not implemented")


if __name__ == "__main__":
    from matplotlib import pyplot as plt
