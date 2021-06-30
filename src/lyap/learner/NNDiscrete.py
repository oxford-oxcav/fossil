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

from src.lyap.verifier.verifier import Verifier
from src.lyap.verifier.z3verifier import Z3Verifier
from src.shared.cegis_values import CegisConfig ,CegisStateKeys
from src.shared.consts import LearningFactors
from src.shared.learner import Learner
from src.shared.activations import ActivationType, activation, activation_der
from src.lyap.utils import Timer, timer, get_symbolic_formula, vprint
from src.shared.sympy_converter import sympy_converter

T = Timer()


class NNDiscrete(nn.Module, Learner):
    def __init__(self, input_size, learn_method, *args, bias=True, activate=ActivationType.LIN_SQUARE, equilibria=0, llo=False, **kw):
        super(NNDiscrete, self).__init__()

        self.input_size = input_size
        n_prev = input_size
        self.eq = equilibria
        self.llo = llo
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
            'Pow': torch.pow,
            'Real': lambda x: x
        }

    # generalisation of forward with tensors
    def forward(self, x):
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

        numerical_v = torch.matmul(y, self.layers[-1].weight.T)

        return numerical_v[:, 0]

    def numerical_net(self, S, Sdot, lf):
        assert (len(S) == len(Sdot))

        nn = self.forward(S)
        nn_next = self.forward(Sdot)
        # circle = x0*x0 + ... + xN*xN
        circle = torch.pow(S, 2).sum(dim=1)

        E = self.compute_factors(S, lf)

        # define E(x) := (x-eq_0) * ... * (x-eq_N)
        # V = NN(x) * E(x)
        V = nn * E
        delta_V = nn_next * E - V

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
                    E *= torch.sum(torch.pow(S-torch.tensor(self.eq[idx, :]), 2), dim=1)
            else:
                E = torch.tensor(1.0)

        return E
    
    def get(self, **kw):
        return self.learn(kw[CegisStateKeys.optimizer], kw[CegisStateKeys.S],
                          kw[CegisStateKeys.S_dot], kw[CegisStateKeys.factors])

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
        min_dist = float('inf')
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
            return np.ceil(np.log10(number))
        else:
            return 1.0

    @staticmethod
    def get_timer():
        return T

