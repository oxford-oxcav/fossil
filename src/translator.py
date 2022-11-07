# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import logging
from typing import Literal, Tuple, Union

import numpy as np
import sympy as sp
import torch

try:
    from maraboupy.Marabou import read_onnx
    from maraboupy.MarabouNetworkONNX import MarabouNetworkONNX
    marabou = True
except ImportError as e:
    logging.exception("Exception while importing Marabou")
    marabou = False


import src.learner as learner
from src.shared.activations_symbolic import activation_der_sym, activation_sym
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.shared.component import Component
from src.shared.consts import LearningFactors, TimeDomain, VerifierType
from src.shared.sympy_converter import sympy_converter
from src.shared.utils import Timer, timer, vprint

T = Timer()


class TranslatorCT(Component):
    def __init__(self, net, x, xdot, eq, rounding, **kw):
        super().__init__()
        self.net = net
        self.x = np.array(x).reshape(-1, 1)
        self.xdot = np.array(xdot).reshape(-1, 1)
        self.eq = eq
        self.round = rounding
        self.verbose = kw.get(CegisConfig.VERBOSE.k, CegisConfig.VERBOSE.v)

    @timer(T)
    def get(self, **kw):
        # to disable rounded numbers, set rounding=-1
        sp_handle = kw.get(CegisStateKeys.sp_handle, False)
        fcts = kw.get(CegisStateKeys.factors)
        self.xdot = np.array(kw.get(CegisStateKeys.xdot, self.xdot)).reshape(-1, 1)
        V, Vdot = self.get_symbolic_formula(self.x, self.xdot, lf=fcts)

        if sp_handle:
            V, Vdot = sp.simplify(V), sp.simplify(Vdot)
            x_map = kw[CegisStateKeys.x_v_map]
            V = sympy_converter(x_map, V)
            Vdot = sympy_converter(x_map, Vdot)

        vprint(["Candidate: {}".format(V)], self.verbose)

        return {CegisStateKeys.V: V, CegisStateKeys.V_dot: Vdot}

    def get_symbolic_formula(self, x, xdot, lf=None):
        """
        :param net:
        :param x:
        :param xdot:
        :return:
        """

        z, jacobian = self.network_until_last_layer(x)

        if self.round < 0:
            last_layer = self.net.layers[-1].weight.data.numpy()
        else:
            last_layer = np.round(self.net.layers[-1].weight.data.numpy(), self.round)

        z = last_layer @ z
        jacobian = last_layer @ jacobian  # jacobian now contains the grad V

        assert z.shape == (1, 1)
        # V = NN(x) * E(x)
        E, derivative_e = self.compute_factors(np.array(x).reshape(1, -1), lf)

        # gradV = der(NN) * E + dE/dx * NN
        gradV = np.multiply(jacobian, np.broadcast_to(E, jacobian.shape)) + np.multiply(
            derivative_e, np.broadcast_to(z[0, 0], jacobian.shape)
        )
        # Vdot = gradV * f(x)
        Vdot = gradV @ xdot

        if isinstance(E, sp.Add):
            V = sp.expand(z[0, 0] * E)
            Vdot = sp.expand(Vdot[0, 0])
        else:
            V = z[0, 0] * E
            Vdot = Vdot[0, 0]

        return V, Vdot

    def network_until_last_layer(self, x):
        """
        :param x:
        :return:
        """
        z = x
        jacobian = np.eye(self.net.input_size, self.net.input_size)

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
            z = activation_sym(self.net.acts[idx], zhat)
            # Vdot
            jacobian = w @ jacobian
            jacobian = (
                np.diagflat(activation_der_sym(self.net.acts[idx], zhat)) @ jacobian
            )

        return z, jacobian

    def compute_factors(self, x, lf):
        """
        :param x:
        :param lf: linear factors
        :return:
        """
        if lf == LearningFactors.QUADRATIC:  # quadratic terms
            E, temp = 1, []
            factors = np.full(
                shape=(self.eq.shape[0], x.shape[0]), dtype=object, fill_value=0
            )
            for idx in range(self.eq.shape[0]):  # number of equilibrium points
                E *= sum(np.power((x.T - self.eq[idx, :].reshape(x.T.shape)), 2).T)[
                    0, 0
                ]
                factors[idx] = x.T - self.eq[idx, :].reshape(x.T.shape)
            # derivative = 2*(x-eq)*E/E_i
            grad_e = sp.zeros(1, x.shape[0])
            for var in range(x.shape[0]):
                for idx in range(self.eq.shape[0]):
                    grad_e[var] += sp.simplify(
                        E
                        * factors[idx, var]
                        / sum(
                            np.power((x.T - self.eq[idx, :].reshape(x.T.shape)), 2).T
                        )[0, 0]
                    )
            derivative_e = 2 * grad_e
        else:  # no factors
            E, derivative_e = 1.0, 0.0

        return E, derivative_e

    @staticmethod
    def get_timer():
        return T


class TranslatorDT(TranslatorCT):
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

        vprint(["Candidate: {}".format(V)], self.verbose)

        return {CegisStateKeys.V: V, CegisStateKeys.V_dot: Vdot}

    def get_symbolic_formula(self, x, xdot, lf=None):
        """
        :param net:
        :param x:
        :param xdot:
        :return:
        """

        z, z_xdot = self.network_until_last_layer(x), self.network_until_last_layer(
            xdot
        )

        if self.round < 0:
            last_layer = self.net.layers[-1].weight.data.numpy()
        else:
            last_layer = np.round(self.net.layers[-1].weight.data.numpy(), self.round)

        z = last_layer @ z
        z_xdot = last_layer @ z_xdot

        assert z.shape == (1, 1)
        # V = NN(x) * E(x)
        E = self.compute_factors(np.array(x).reshape(1, -1), lf)

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
            z = activation_sym(self.net.acts[idx], zhat)
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
            factors = np.full(
                shape=(self.eq.shape[0], x.shape[0]), dtype=object, fill_value=0
            )
            for idx in range(self.eq.shape[0]):  # number of equilibrium points
                E *= sum(np.power((x.T - self.eq[idx, :].reshape(x.T.shape)), 2).T)[
                    0, 0
                ]
                factors[idx] = x.T - self.eq[idx, :].reshape(x.T.shape)
            # derivative = 2*(x-eq)*E/E_i
        else:  # no factors
            E = 1.0

        return E

    @staticmethod
    def get_timer():
        return T


if marabou:

    class _DiffNet(torch.nn.Module):
        """Private class to provide forward method of delta_V for Marabou

        V (NNDiscrete): Candidate Lyapunov ReluNet
        f (EstimNet): Estimate of system dynamics ReluNet
        """

        def __init__(self, V: learner.LearnerDT, f) -> None:
            super(_DiffNet, self).__init__()
            self.V = V
            # Means forward can only be called with batchsize = 1
            self.factor = torch.nn.Parameter(-1 * torch.ones([1, 1]))
            self.F = f

        def forward(self, S, Sdot) -> torch.Tensor:
            return self.V(self.F(S), self.F(Sdot))[0] + self.factor @ self.V(S, Sdot)[0]


    class MarabouTranslator(Component):
        """Takes an torch nn.module object and converts it to an onnx file to be read by marabou

        dimension (int): Dimension of dynamical system
        """

        def __init__(self, dimension: int):
            self.dimension = dimension

        @timer(T)
        def get(
            self, net: learner.LearnerDT = None, ENet=None, **kw
        ) -> Tuple[MarabouNetworkONNX, MarabouNetworkONNX]:
            """
            net (NNDiscrete): PyTorch candidate Lyapunov Neural Network
            ENet (EstimNet): dynamical system as PyTorch Neural Network
            """
            tf_V = tempfile.NamedTemporaryFile(suffix=".onnx")
            tf_DV = tempfile.NamedTemporaryFile(suffix=".onnx")
            model = _DiffNet(net, ENet)
            self.export_net_to_file(net, tf_V, "V")
            self.export_net_to_file(model, tf_DV, "dV")

            V_net = read_onnx(tf_V.name, outputName="V")
            dV_net = read_onnx(tf_DV.name, outputName="dV")
            return {CegisStateKeys.V: V_net, CegisStateKeys.V_dot: dV_net}

        def export_net_to_file(
            self, net: Union[_DiffNet, learner.LearnerDT], tf, output: str
        ) -> None:
            dummy_input = (torch.rand([1, self.dimension]), torch.rand([1, self.dimension]))
            torch.onnx.export(
                net,
                dummy_input,
                tf,
                input_names=["S", "Sdot"],
                output_names=[output],
                opset_version=11,
            )

    @staticmethod
    def get_timer():
        return T


def get_translator_type(time_domain: Literal, verifier: Literal) -> Component:
    if verifier == VerifierType.MARABOU:
        if time_domain != TimeDomain.DISCRETE:
            raise ValueError(
                "Marabou verifier not compatible with continuous-time dynamics"
            )
        return MarabouTranslator
    elif time_domain == TimeDomain.DISCRETE:
        return TranslatorDT
    elif time_domain == TimeDomain.CONTINUOUS:
        return TranslatorCT
    else:
        TypeError("Not Implemented Translator")


def get_translator(translator_type: Component, net, x, xdot, eq, rounding, **kw):
    if translator_type == TranslatorCT or translator_type == TranslatorDT:
        return translator_type(net, x, xdot, eq, rounding, **kw)
    elif translator_type == MarabouTranslator:
        return translator_type(x.shape[0])
