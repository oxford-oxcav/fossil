# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
# pylint: disable=not-callable
import torch
from src.shared.cegis_values import CegisStateKeys
from src.translator.translator_continuous import TranslatorContinuous
from src.shared.consts import LearningFactors
from src.shared.sympy_converter import sympy_converter
from src.shared.utils import *
import sympy as sp
import numpy as np

T = Timer()


class TranslatorDiscrete(TranslatorContinuous):
    def __init__(self, net, x, xdot, eq, rounding, **kw):
        super().__init__(net, x, xdot, eq, rounding, **kw)

    @timer(T)
    def get(self, **kw):
        # to disable rounded numbers, set rounding=-1
        sp_handle = kw.get(CegisStateKeys.sp_handle, False)
        fcts = kw.get(CegisStateKeys.factors)

        V, Vdot = self.get_symbolic_formula(self.x, self.xdot, lf=fcts)

        if sp_handle:
            V, Vdot = sp.simplify(V), sp.simplify(Vdot)
            x_map = kw[CegisStateKeys.x_v_map]
            V = sympy_converter(x_map, V)
            Vdot = sympy_converter(x_map, Vdot)

        vprint(['Candidate: {}'.format(V)], self.verbose)

        return {CegisStateKeys.V: V, CegisStateKeys.V_dot: Vdot}

    def get_symbolic_formula(self, x, xdot, lf=None):
        """
        :param net:
        :param x:
        :param xdot:
        :return:
        """

        z, z_xdot = self.network_until_last_layer(x), self.network_until_last_layer(xdot)

        if self.round < 0:
            last_layer = self.net.layers[-1].weight.data.numpy()
        else:
            last_layer = np.round(self.net.layers[-1].weight.data.numpy(), self.round)

        z = last_layer @ z
        z_xdot = last_layer @ z_xdot  

        assert z.shape == (1, 1)
        # V = NN(x) * E(x)
        E = self.compute_factors(np.array(x).reshape(1,-1), lf)

        # gradV = der(NN) * E + dE/dx * NN
        
        if isinstance(E, sp.Add):
            V = sp.expand(z[0, 0] * E)
            z_xdot = sp.expand(z_xdot[0, 0] * E)
        else:
            V = z[0, 0] * E
            z_xdot = z_xdot[0, 0]

        return V, z_xdot - V

    def network_until_last_layer(self, x):
        """
        :param x:
        :return:
        """
        z = x

        for idx, layer in enumerate(self.net.layers[:-1]):
            if self.round < 0:
                w = layer.weight.data.numpy()
                if layer.bias is not None:
                    b = layer.bias.data.numpy()[:, None]
                else:
                    b = np.zeros((layer.out_features, 1))
            elif self.round > 0:
                w = np.round(layer.weight.data.numpy(), self.round)
                if layer.bias is not None:
                    b = np.round(layer.bias.data.numpy(), self.round)[:, None]
                else:
                    b = np.zeros((layer.out_features, 1))

            zhat = w @ z + b
            z = activation_z3(self.net.acts[idx], zhat)
            # Vdot
        return z

    def compute_factors(self, x, lf):
        """
        :param x:
        :param lf: linear factors
        :return:
        """
        if lf == LearningFactors.QUADRATIC:  # quadratic terms
            E, temp = 1, []
            factors = np.full(shape=(self.eq.shape[0], x.shape[0]), dtype=object, fill_value=0)
            for idx in range(self.eq.shape[0]):  # number of equilibrium points
                E *= sum(np.power((x.T - self.eq[idx, :].reshape(x.T.shape)), 2).T)[0, 0]
                factors[idx] = (x.T - self.eq[idx, :].reshape(x.T.shape))
            # derivative = 2*(x-eq)*E/E_i
        else:  # no factors
            E = 1.0

        return E

    @staticmethod
    def get_timer():
        return T

