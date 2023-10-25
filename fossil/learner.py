# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Literal
import warnings

import numpy as np
import torch
import torch.nn as nn

from fossil.activations import activation
from fossil.component import Component
from fossil.consts import *
from fossil.utils import Timer, timer

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


class LearnerNN(nn.Module, Learner):
    def __init__(
        self,
        input_size,
        learn_method,
        *args,
        activation: tuple[ActivationType, ...] = (ActivationType.SQUARE,),
        config: CegisConfig = CegisConfig(),
        bias=True,
    ):
        super(LearnerNN, self).__init__()

        self.input_size = input_size
        n_prev = self.input_size
        self._diagonalise = False
        self.acts = activation
        self._is_there_bias = bias
        self.verbose = config.VERBOSE
        ZaZ = config.FACTORS  # kw.get(CegisConfig.FACTORS.k, CegisConfig.FACTORS.v)
        self.factor = QuadraticFactor() if ZaZ == LearningFactors.QUADRATIC else None
        self.layers = []
        self._take_abs = config.LLO and not self.is_final_polynomial()
        self.beta = None
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
        layer = nn.Linear(n_prev, 1, bias=bias)
        # last layer of ones
        if config.LLO and not self._take_abs:
            layer.weight = torch.nn.Parameter(torch.ones(layer.weight.shape))
            self.layers.append(layer)
        else:  # free output layer
            self.register_parameter("W" + str(k), layer.weight)
            if bias:
                self.register_parameter("b" + str(k), layer.bias)
            self.layers.append(layer)
        if config.LLO and not self.is_positive_definite():
            warnings.warn("LLO set but function is not positive definite")
        self.learn_method = learn_method
        self._type = config.CERTIFICATE.name

    # backprop algo
    @timer(T)
    def learn(
        self,
        net: "LearnerNN",
        optimizer: torch.optim.Optimizer,
        S: torch.Tensor,
        Sdot: torch.Tensor,
        xdot_func: Callable,
    ) -> dict:
        return self.learn_method(net, optimizer, S, Sdot, xdot_func)

    def get(self, **kw):
        return self.learn(
            kw[CegisStateKeys.net],
            kw[CegisStateKeys.optimizer],
            kw[CegisStateKeys.S],
            kw[CegisStateKeys.S_dot],
            None,
            # I think this could actually still pass xdot_func, since there's no pytorch parameters to learn
        )

    def make_final_layer_positive(self):
        """Makes the last layer of the neural network positive definite."""
        with torch.no_grad():
            self.layers[-1].weight.data = torch.abs(self.layers[-1].weight.data)

    def get_all(
        self, S: torch.Tensor, Sdot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the value of the learner, its lie derivative and the circle."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass of the neural network.

            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        y = x

        for idx, layer in enumerate(self.layers[:-1]):
            z = layer(y)
            y = activation(self.acts[idx], z)

        y = self.layers[-1](y)[:, 0]
        return y

    def freeze(self):
        """Freezes the parameters of the neural network by setting requires_grad to False."""

        for param in self.parameters():
            if not param.requires_grad:
                break
            param.requires_grad = False

    def compute_net_gradnet(self, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the value of the neural network and its gradient.

        Computes gradient using autograd.

            S (torch.Tensor): input tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (nn, grad_nn)
        """
        S_clone = torch.clone(S).requires_grad_()
        nn = self(S_clone)

        grad_nn = torch.autograd.grad(
            outputs=nn,
            inputs=S_clone,
            grad_outputs=torch.ones_like(nn),
            create_graph=True,
            retain_graph=True,
            # allow_unused=True,
        )[0]
        return nn, grad_nn

    def compute_V_gradV(
        self, nn: torch.Tensor, grad_nn: torch.Tensor, S: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the value of the function and its gradient.

        The function is defined as:
            V = NN(x) * F(x)
        where NN(x) is the neural network and F(x) is a factor, equal to either 1 or ||x||^2.

            nn (torch.Tensor): neural network value
            grad_nn (torch.Tensor): gradient of the neural network
            S (torch.Tensor): input tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (V, gradV)
        """
        F, derivative_F = self.compute_factors(S)
        V = nn * F
        # define F(x) := ||x||^2
        # V = NN(x) * F(x)
        # gradV = NN(x) * dF(x)/dx  + der(NN) * F(x)
        # gradV = torch.stack([nn, nn]).T * derivative_e + grad_nn * torch.stack([E, E]).T
        if self.factor is not None:
            gradV = (
                nn.expand_as(grad_nn.T).T * derivative_F.expand_as(grad_nn)
                + grad_nn * F.expand_as(grad_nn.T).T
            )
        else:
            gradV = grad_nn
        return V, gradV

    @staticmethod
    def learner_fncts():
        return {
            "sin": torch.sin,
            "cos": torch.cos,
            "exp": torch.exp,
            "If": lambda cond, _then, _else: _then if cond.item() else _else,
        }

    def compute_factors(self, S: torch.Tensor):
        if self.factor:
            return self.factor(S), self.factor.derivative(S)
        else:
            return 1, 0

    def compute_minimum(self, S: torch.Tensor) -> tuple[float, float]:
        """Computes the minimum of the learner over the input set.

        Also returns the argmin of the minimum.

        Args:
            S (torch.Tensor): _description_

        Returns:
            tuple[float, float]: _description_
        """
        C = self(S)
        minimum = torch.min(C, 0)
        value = minimum.values.item()
        index = minimum.indices.item()
        argmin = S[index]
        return value, argmin

    def compute_maximum(self, S: torch.Tensor) -> tuple[float, float]:
        """Computes the maximum of the learner over the input set.

        Also returns the argmax of the maximum.

        Args:
            S (torch.Tensor): _description_

        Returns:
            tuple[float, float]: _description_
        """
        C = self(S)
        maximum = torch.max(C, 0)
        value = maximum.values.item()
        index = maximum.indices.item()
        argmax = S[index]
        return value, argmax

    def find_closest_unsat(self, S, Sdot):
        min_dist = float("inf")
        V, Vdot, _ = self.get_all(S, Sdot)
        for iii in range(S[0].shape[0]):
            if V[iii].item() < 0 or Vdot[iii].item() > 0:
                dist = S[0][iii].norm()
                if dist < min_dist:
                    min_dist = dist
        self.closest_unsat = min_dist

    def diagonalisation(self):
        # makes the weight matrices diagonal. works iff intermediate layers are square matrices
        with torch.no_grad():
            for layer in self.layers[:-1]:
                layer.weight.data = torch.diag(torch.diag(layer.weight))

    def is_positive_definite(self):
        """Checks if the net is positive (semi) definite, assuming W is also positive definite;

        Positive definiteness is defined as the following:
        N(x) > 0 for all x in R^n


        Returns:
            bool: True is net is positive definite, else False
        """
        activations = self.acts
        layers = self.layers
        pd_acts = [
            ActivationType.RELU,
            ActivationType.SOFTPLUS,
            ActivationType.SQUARE,
            ActivationType.COSH,
            ActivationType.SIGMOID,
            ActivationType.EVEN_POLY_4,
            ActivationType.EVEN_POLY_6,
            ActivationType.EVEN_POLY_8,
            ActivationType.EVEN_POLY_10,
        ]
        return (activations[-1] in pd_acts) and (layers[-1].bias is None)

    def is_final_polynomial(self):
        """Checks if the final layer is a polynomial"""
        act = self.acts[-1].name
        # Shortcut rather than listing all polynomial activations
        poly_names = "POLY", "SQUARE"
        return any([poly_name in act for poly_name in poly_names])

    def clean(self):
        """Prepares object for pickling by removing unpicklable attributes."""
        self.learn_method = None

    @staticmethod
    def order_of_magnitude(number):
        if number.item() != 0:
            return np.ceil(np.log10(number))
        else:
            return 1.0

    @staticmethod
    def get_timer():
        return T

    def compute_dV(self, gradV, Sdot):
        raise NotImplementedError("compute_dV not implemented for this learner")


class LearnerCT(LearnerNN):
    """Leaner class for continuous time dynamical models.

    Learns and evaluates V and Vdot.

    """

    def get_all(
        self, S: torch.Tensor, Sdot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the value of the function, its lie derivative and circle.

        Assumes CT model.
        The function is defined as:
            V = NN(x) * F(x)
        where NN(x) is the neural network and F(x) is a factor, equal to either 1 or ||x||^2.
            S (torch.Tensor): Samples over the domain.
            Sdot (torch.Tensor): Dynamical model evaluated at S.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                V (torch.Tensor): Value of the function.
                Vdot (torch.Tensor): Lie derivative of the function.
                circle (torch.Tensor): Circle of the function.
        """
        assert len(S) == len(Sdot)

        nn, grad_nn = self.compute_net_gradnet(S)
        # circle = x0*x0 + ... + xN*xN
        circle = torch.pow(S, 2).sum(dim=1)

        V, gradV = self.compute_V_gradV(nn, grad_nn, S)
        Vdot = self.compute_dV(gradV, Sdot)

        return V, Vdot, circle

    def compute_dV(self, gradV: torch.Tensor, Sdot: torch.Tensor) -> torch.Tensor:
        """Computes the  lie derivative of the function.

        Args:
            gradV (torch.Tensor): gradient of the function
            Sdot (torch.Tensor): df/dt

        Returns:
            torch.Tensor: dV/dt
        """
        # Vdot = gradV * f(x)
        Vdot = torch.sum(torch.mul(gradV, Sdot), dim=1)
        return Vdot


class LearnerDT(LearnerNN):
    """Leaner class for discrete time dynamical models."""

    def get_all(
        self, S: torch.Tensor, Sdot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes V, delta_V and circle.

        Args:
            S (torch.Tensor): samples over the domain
            Sdot (torch.Tensor): f(x) evaluated at the samples

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                V (torch.Tensor): Value of the function.
                delta V (torch.Tensor): One step difference of the function.
                circle (torch.Tensor): Circle of the function.
        """
        # assert (len(S) == len(Sdot))  ## This causes a warning in Marabou

        nn = self.forward(S)
        nn_next = self.forward(Sdot)
        # circle = x0*x0 + ... + xN*xN
        circle = torch.pow(S, 2).sum(dim=1)

        E = self.compute_factors(S)

        # define E(x) := (x-eq_0) * ... * (x-eq_N)
        # V = NN(x) * E(x)
        V = nn
        delta_V = nn_next - V

        return V, delta_V, circle


class CtrlLearnerCT(LearnerCT):
    def __init__(
        self,
        input_size,
        learn_method,
        *args,
        activation: tuple[ActivationType, ...] = (ActivationType.SQUARE,),
        bias=True,
        config: CegisConfig = CegisConfig(),
    ):
        LearnerNN.__init__(
            self,
            input_size,
            learn_method,
            *args,
            activation=activation,
            bias=bias,
            config=config,
        )
        self.ctrl_layers = config.CTRLAYER

        # if self.ctrl_layers is not None:
        #     self.ctrler = GeneralController(inputs=input_size, output=self.ctrl_layers[-1],
        #                                     layers=self.ctrl_layers[:-1],
        #                                     activations=[ActivationType.LINEAR]*len(self.ctrl_layers))

    def get(self, **kw):
        return self.learn(
            kw[CegisStateKeys.net],
            kw[CegisStateKeys.optimizer],
            kw[CegisStateKeys.S],
            kw[CegisStateKeys.S_dot],
            kw[CegisStateKeys.xdot_func],
        )

    # backprop algo
    @timer(T)
    def learn(self, net, optimizer, S, Sdot, xdot_func):
        return self.learn_method(net, optimizer, S, Sdot, xdot_func)


class CtrlLearnerDT(LearnerDT):
    def __init__(
        self,
        input_size,
        learn_method,
        *args,
        activation: tuple[ActivationType, ...] = (ActivationType.SQUARE,),
        bias=True,
        config: CegisConfig = CegisConfig(),
    ):
        LearnerNN.__init__(
            self,
            input_size,
            learn_method,
            *args,
            activation=activation,
            bias=bias,
            config=config,
        )
        self.ctrl_layers = config.CTRLAYER
        # if self.ctrl_layers is not None:
        #     self.ctrler = GeneralController(inputs=input_size, output=self.ctrl_layers[-1],
        #                                     layers=self.ctrl_layers[:-1],
        #                                     activations=[ActivationType.LINEAR]*len(self.ctrl_layers))

    def get(self, **kw):
        return self.learn(
            kw[CegisStateKeys.net],
            kw[CegisStateKeys.optimizer],
            kw[CegisStateKeys.S],
            kw[CegisStateKeys.S_dot],
            kw[CegisStateKeys.xdot_func],
        )

    # backprop algo
    @timer(T)
    def learn(self, net, optimizer, S, Sdot, xdot_func):
        return self.learn_method(net, optimizer, S, Sdot, xdot_func)


def get_learner(time_domain: Literal, ctrl: Literal) -> LearnerNN:
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
