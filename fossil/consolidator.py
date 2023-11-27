# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import torch

from fossil.consts import CegisStateKeys
from fossil.component import Component
from fossil.activations import activation, activation_der
from fossil.utils import Timer, timer

T = Timer()


class Consolidator(Component):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def get(self, **kw):
        for label, cex in kw[CegisStateKeys.cex].items():
            if (
                "lie" in label and cex != []
            ):  # Trying to 'generalise' when we use the trajectoriser
                return self.compute_trajectory(kw[CegisStateKeys.net], cex[-1])
        return {CegisStateKeys.trajectory: []}

    # computes the gradient of V, Vdot in point
    # computes a 20-step trajectory (20 is arbitrary) starting from point
    # towards increase: + gamma*grad
    # towards decrease: - gamma*grad
    @timer(T)
    def compute_trajectory(self, net, point):
        """
        :param net: NN object
        :param point: tensor
        :return: list of tensors
        """
        # set some parameters
        gamma = 0.05  # step-size factor
        max_iters = 20
        # fixing possible dimensionality issues
        point.requires_grad = True
        trajectory = [point]
        num_vdot_value_old = -1.0
        # gradient computation
        for gradient_loop in range(max_iters):
            # compute gradient of Vdot
            gradient, num_vdot_value = self.compute_Vdot_grad(net, point)
            # set break conditions
            if (
                num_vdot_value_old > num_vdot_value
                or abs(num_vdot_value_old - num_vdot_value) < 1e-5
                or num_vdot_value > 1e6
                or (abs(gradient) > 1e2).any()
            ):
                break
            else:
                num_vdot_value_old = num_vdot_value
            # "detach" and "requires_grad" make the new point "forget" about previous operations
            point = point.clone().detach() + gamma * gradient.clone().detach()
            point.requires_grad = True
            trajectory.append(point)
        # just checking if gradient is numerically unstable
        assert not torch.isnan(torch.stack(trajectory)).any()
        return {CegisStateKeys.trajectory: torch.stack(trajectory)}

    def compute_Vdot_grad(self, net, point):
        """
        :param net:
        :param point:
        :return:
        """
        num_v_dot = self.forward_Vdot(net, point)
        num_v_dot.backward()
        grad_v_dot = point.grad
        assert grad_v_dot is not None
        return grad_v_dot, num_v_dot

    def forward_Vdot(self, net, x):
        """
        :param x: tensor of data points
        :param xdot: tensor of data points
        :return:
                Vdot: tensor, evaluation of x in derivative net
        """
        y = x[None, :]
        xdot = self.f(y)

        _, Vdot, _ = net.get_all(y, xdot)

        return Vdot

    @staticmethod
    def get_timer():
        return T


class DoubleConsolidator(Consolidator):
    def __init__(self, f):
        super().__init__(f)

    def get(self, index, **kw):
        net = kw[CegisStateKeys.net][index]
        for label, cex in kw[CegisStateKeys.cex].items():
            if (
                "lie" in label and cex != []
            ):  # Trying to 'generalise' when we use the trajectoriser
                return self.compute_trajectory(net, cex[-1])
        return {CegisStateKeys.trajectory: []}
