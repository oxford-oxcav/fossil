# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import torch
import torch.nn as nn
import numpy as np
import sympy as sp
import z3

from src.shared.cegis_values import CegisConfig ,CegisStateKeys
from src.shared.consts import LearningFactors
from src.learner.learner import Learner
from src.shared.activations import ActivationType, activation, activation_der
from src.shared.utils import Timer, timer

T = Timer()


class NNContinuous(nn.Module, Learner):
    def __init__(self, input_size, learn_method, *args, bias=True, activate=ActivationType.SQUARE, equilibria=0, llo=False, **kw):
        super(NNContinuous, self).__init__()

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
        else:          # free output layer
            self.register_parameter("W" + str(k), layer.weight)
            self.layers.append(layer)
        self.learn_method = learn_method

    @staticmethod
    def learner_fncts():
        return {
            'sin': torch.sin,
            'cos': torch.cos,
            'exp': torch.exp,
            'If': lambda cond, _then, _else: _then if cond.item() else _else,
        }

    def forward(self, S, Sdot):
        assert (len(S) == len(Sdot))

        nn, grad_nn = self.forward_tensors(S)
        # circle = x0*x0 + ... + xN*xN
        circle = torch.pow(S, 2).sum(dim=1)

        E, derivative_e = self.compute_factors(S, self.factors)

        # define E(x) := (x-eq_0) * ... * (x-eq_N)
        # V = NN(x) * E(x)
        V = nn * E
        # gradV = NN(x) * dE(x)/dx  + der(NN) * E(x)
        # gradV = torch.stack([nn, nn]).T * derivative_e + grad_nn * torch.stack([E, E]).T
        gradV = nn.expand_as(grad_nn.T).T * derivative_e.expand_as(grad_nn) \
                + grad_nn * E.expand_as(grad_nn.T).T
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
            jacobian = torch.matmul(torch.diag_embed(activation_der(self.acts[idx], z)), jacobian)

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
                    factors[:, :, idx] = S-torch.tensor(self.eq[idx, :])
                    E *= torch.sum(torch.pow(S-torch.tensor(self.eq[idx, :]), 2), dim=1)

                # derivative = 2*(x-eq)*E/E_i
                grad_e = torch.zeros(S.shape[0], self.input_size)
                for var in range(self.input_size):
                    for idx in range(self.eq.shape[0]):
                        grad_e[:, var] += \
                            E * factors[:, var, idx] / torch.sum(torch.pow(S-torch.tensor(self.eq[idx, :]), 2), dim=1)
                derivative_e = 2*grad_e
            else:
                E, derivative_e = torch.tensor(1.0), torch.tensor(0.0)

        return E, derivative_e
    
    def get(self, **kw):
        return self.learn(kw[CegisStateKeys.optimizer], kw[CegisStateKeys.S],
                          kw[CegisStateKeys.S_dot])

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
        min_dist = float('inf')
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
