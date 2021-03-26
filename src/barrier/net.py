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

from src.barrier.verifier import Verifier
from src.shared.activations import ActivationType, activation, activation_der
from src.barrier.utils import Timer, timer, get_symbolic_formula, vprint
from src.shared.cegis_values import CegisStateKeys, CegisConfig
from src.shared.learner import Learner
from src.shared.sympy_converter import sympy_converter

T = Timer()


class NN(nn.Module, Learner):
    def __init__(self, input_size, *args, activate=ActivationType.SQUARE, bias=False, **kw):
        super(NN, self).__init__()

        self.symmetric_belt = kw.get(CegisConfig.SYMMETRIC_BELT.k, CegisConfig.SYMMETRIC_BELT.v)
        self.input_size = input_size
        n_prev = input_size
        self.equilibrium = torch.zeros((1, self.input_size))
        self.acts = activate
        self._is_there_bias = bias
        self.verbose = kw.get(CegisConfig.VERBOSE.k, CegisConfig.VERBOSE.v)
        self.layers = []
        k = 1
        for n_hid in args:
            layer = nn.Linear(n_prev, n_hid, bias=bias)
            self.register_parameter("W" + str(k), layer.weight)
            if bias:
                self.register_parameter("b" + str(k), layer.bias)
            self.layers.append(layer)
            n_prev = n_hid
            k = k + 1

        # free output layer
        layer = nn.Linear(n_prev, 1, bias=False)
        self.register_parameter("W" + str(k), layer.weight)
        self.layers.append(layer)
        self.output_layer = layer.weight.clone().detach()
        # or
        # self.output_layer = torch.ones(1, n_prev)

    @staticmethod
    def learner_fncts():
        return {
            'sin': torch.sin,
            'cos': torch.cos,
            'exp': torch.exp,
            'If': lambda cond, _then, _else: _then if cond.item() else _else,
        }

    # generalisation of forward with tensors
    def forward_tensors(self, x, xdot):
        """
        :param x: tensor of data points
        :param xdot: tensor of data points
        :return:
                V: tensor, evaluation of x in net
                Vdot: tensor, evaluation of x in derivative net
        """
        y = x
        jacobian = torch.diag_embed(torch.ones(x.shape[0], self.input_size))

        for idx, layer in enumerate(self.layers[:-1]):
            z = layer(y)
            y = activation(self.acts[idx], z)

            jacobian = torch.matmul(layer.weight, jacobian)
            jacobian = torch.matmul(torch.diag_embed(activation_der(self.acts[idx], z)), jacobian)

        numerical_b = torch.matmul(y, self.layers[-1].weight.T)
        jacobian = torch.matmul(self.layers[-1].weight, jacobian)
        numerical_bdot = torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1)

        return numerical_b[:, 0], numerical_bdot, y

    def numerical_net(self, S, Sdot):
        """
        :param net: NN object
        :param S: tensor
        :param Sdot: tensor
        :return: V, Vdot, circle: tensors
        """
        assert (len(S) == len(Sdot))

        B, Bdot, _ = self.forward_tensors(S, Sdot)
        # circle = x0*x0 + ... + xN*xN
        circle = torch.pow(S, 2).sum(dim=1)

        return B, Bdot, circle

    def get(self, **kw):
        return self.learn(kw[CegisStateKeys.optimizer], kw[CegisStateKeys.S], kw[CegisStateKeys.S_dot])

    @timer(T)
    def learn(self, optimizer, S, Sdot):
        """
        :param optimizer: torch optimiser
        :param S: tensor of data
        :param Sdot: tensor contain f(data)
        :param margin: performance threshold
        :return: --
        """
        assert (len(S) == len(Sdot))

        learn_loops = 1000
        margin = 0.1
        condition_old = False

        for t in range(learn_loops):
            optimizer.zero_grad()

            # permutation_index = torch.randperm(S[0].size()[0])
            # permuted_S, permuted_Sdot = S[0][permutation_index], S_dot[0][permutation_index]
            B_d, Bdot_d, __ = self.numerical_net(S[0], Sdot[0])
            B_i, _, __ = self.numerical_net(S[1], Sdot[1])
            B_u, _, __ = self.numerical_net(S[2], Sdot[2])

            learn_accuracy = sum(B_i <= -margin).item() + sum(B_u >= margin).item()
            percent_accuracy_init_unsafe = learn_accuracy * 100 / (len(S[1]) + len(S[2]))
            percent_accuracy = percent_accuracy_init_unsafe
            slope = 1 / 10 ** 4  # (self.orderOfMagnitude(max(abs(Vdot)).detach()))
            leaky_relu = torch.nn.LeakyReLU(slope)
            relu6 = torch.nn.ReLU6()
            # saturated_leaky_relu = torch.nn.ReLU6() - 0.01*torch.relu()
            loss = (torch.relu(B_i + margin) - slope*relu6(-B_i + margin)).mean() \
                    + (torch.relu(-B_u + margin) - slope*relu6(B_u + margin)).mean()

            # set two belts
            percent_belt = 0
            if self.symmetric_belt:
                belt_index = torch.nonzero(torch.abs(B_d) <= 0.5)
            else:
                belt_index = torch.nonzero(B_d >= -margin)

            if belt_index.nelement() != 0:
                dB_belt = torch.index_select(Bdot_d, dim=0, index=belt_index[:, 0])
                learn_accuracy = learn_accuracy + (sum(dB_belt <= -margin)).item()
                percent_accuracy = 100 * learn_accuracy / (len(S[1]) + len(S[2]) + dB_belt.shape[0])
                percent_belt = 100*(sum(dB_belt <= -margin)).item() / dB_belt.shape[0]

                loss = loss - (relu6(-dB_belt + 0*margin)).mean()

            # loss = loss + (100-percent_accuracy)

            if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                vprint((t, "- loss:", loss.item(), '- accuracy init-unsafe:', percent_accuracy_init_unsafe,
                        "- accuracy belt:", percent_belt, '- points in belt:', len(belt_index)), self.verbose)

            # if learn_accuracy / batch_size > 0.99:
            #     for k in range(batch_size):
            #         if Vdot[k] > -margin:
            #             print("Vdot" + str(S[k].tolist()) + " = " + str(Vdot[k].tolist()))

            if percent_accuracy_init_unsafe == 100 and percent_belt >= 99.9:
                condition = True
            else:
                condition = False

            if condition and condition_old:
                break
            condition_old = condition

            loss.backward()
            optimizer.step()

        return {}

    def to_next_component(self, out, component, **kw):
        if isinstance(component, Verifier):
            sp_handle = kw.get(CegisStateKeys.sp_handle, CegisConfig.SP_HANDLE.v)
            sp_simplify = kw.get(CegisStateKeys.sp_simplify, CegisConfig.SP_SIMPLIFY.v)
            x, xdot = kw[CegisStateKeys.x_v], kw[CegisStateKeys.x_v_dot]
            x_sympy, xdot_s = kw[CegisStateKeys.x_sympy], kw[CegisStateKeys.x_dot_sympy]
            if not sp_handle:  # z3 does all the handling
                B, Bdot = get_symbolic_formula(out, x, xdot)
                if isinstance(B, z3.ArithRef):
                    B, Bdot = z3.simplify(B), z3.simplify(Bdot)
                B_s, Bdot_s = B, Bdot
            else:
                B_s, Bdot_s = get_symbolic_formula(out, x_sympy, xdot_s)

            if sp_simplify:
                B_s = sp.simplify(B_s)
                Bdot_s = sp.simplify(Bdot_s)
            if sp_handle:
                x_map = kw[CegisStateKeys.x_v_map]
                B = sympy_converter(x_map, B_s)
                Bdot = sympy_converter(x_map, Bdot_s)
            return {CegisStateKeys.B: B, CegisStateKeys.B_dot: Bdot}
        return out

    # todo: mv to utils
    def orderOfMagnitude(self, number):
        return np.floor(np.log10(number))

    @staticmethod
    def get_timer():
        return T
