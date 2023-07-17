# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# definition of various activation fcns

import torch

from src import consts


# Activation function
def activation(select: consts.ActivationType, p):
    """
    :param select: enum selects the type of activation
    :param p: the layer
    :return: calls the activation fcn and returns the layer after activation
    """
    if select == consts.ActivationType.IDENTITY:
        return identity(p)
    elif select == consts.ActivationType.RELU:
        return relu(p)
    elif select == consts.ActivationType.LINEAR:
        return p
    elif select == consts.ActivationType.SQUARE:
        return square(p)
    elif select == consts.ActivationType.LIN_SQUARE:
        return lin_square(p)
    elif select == consts.ActivationType.RELU_SQUARE:
        return relu_square(p)
    elif select == consts.ActivationType.REQU:
        return requ(p)
    elif select == consts.ActivationType.TANH:
        return hyper_tan(p)
    elif select == consts.ActivationType.SIGMOID:
        return sigm(p)
    elif select == consts.ActivationType.SOFTPLUS:
        return softplus(p)
    elif select == consts.ActivationType.COSH:
        return cosh(p)
    elif select == consts.ActivationType.LIN_TO_CUBIC:
        return lqc(p)
    elif select == consts.ActivationType.LIN_TO_QUARTIC:
        return lqcq(p)
    elif select == consts.ActivationType.LIN_TO_QUINTIC:
        return lqcqp(p)
    elif select == consts.ActivationType.LIN_TO_SEXTIC:
        return l_e(p)
    elif select == consts.ActivationType.LIN_TO_SEPTIC:
        return l_s(p)
    elif select == consts.ActivationType.LIN_TO_OCTIC:
        return l_o(p)
    elif select == consts.ActivationType.SQUARE_DEC:
        return s_d(p)
    elif select == consts.ActivationType.RATIONAL:
        return rational(p)


def activation_der(select: consts.ActivationType, p):
    """
    :param select: enum selects the type of activation
    :param p: the layer
    :return: calls the activation fcn and returns the layer after activation
    """
    if select == consts.ActivationType.IDENTITY:
        return identity_der(p)
    elif select == consts.ActivationType.RELU:
        return step(p)
    elif select == consts.ActivationType.LINEAR:
        return torch.ones(p.shape)
    elif select == consts.ActivationType.SQUARE:
        return 2 * p
    elif select == consts.ActivationType.LIN_SQUARE:
        return lin_square_der(p)
    elif select == consts.ActivationType.RELU_SQUARE:
        return relu_square_der(p)
    elif select == consts.ActivationType.REQU:
        return 2 * relu(p)
    elif select == consts.ActivationType.TANH:
        return hyper_tan_der(p)
    elif select == consts.ActivationType.SIGMOID:
        return sigm_der(p)
    elif select == consts.ActivationType.SOFTPLUS:
        return softplus_der(p)
    elif select == consts.ActivationType.COSH:
        return sinh(p)
    elif select == consts.ActivationType.LIN_TO_CUBIC:
        return lqc_der(p)
    elif select == consts.ActivationType.LIN_TO_QUARTIC:
        return lqcq_der(p)
    elif select == consts.ActivationType.LIN_TO_QUINTIC:
        return lqcqp_der(p)
    elif select == consts.ActivationType.LIN_TO_SEXTIC:
        return l_e_der(p)
    elif select == consts.ActivationType.LIN_TO_SEPTIC:
        return l_s_der(p)
    elif select == consts.ActivationType.LIN_TO_OCTIC:
        return l_o_der(p)
    elif select == consts.ActivationType.SQUARE_DEC:
        return s_d_der(p)
    elif select == consts.ActivationType.RATIONAL:
        return rational_der(p)


##################################################################
# ACTIVATIONS
##################################################################


def identity(x):
    return x


def relu(x):
    return torch.relu(x)


def square(x):
    return torch.pow(x, 2)


def lin_square(x):
    h = int(x.shape[1] / 2)
    x1, x2 = x[:, :h], x[:, h:]
    return torch.cat([x1, torch.pow(x2, 2)], dim=1)


def relu_square(x):
    h = int(len(x) / 2)
    x1, x2 = x[:h], x[h:]
    return torch.cat([torch.relu(x1), torch.pow(x2, 2)])  # torch.pow(x, 2)


def lqc(x):
    # linear - quadratic - cubic activation
    h = int(x.shape[1] / 3)
    x1, x2, x3 = x[:, :h], x[:, h : 2 * h], x[:, 2 * h :]
    return torch.cat([x1, torch.pow(x2, 2), torch.pow(x3, 3)], dim=1)


def lqcq(x):
    # # linear - quadratic - cubic - quartic activation
    h = int(x.shape[1] / 4)
    x1, x2, x3, x4 = x[:, :h], x[:, h : 2 * h], x[:, 2 * h : 3 * h], x[:, 3 * h :]
    return torch.cat([x1, torch.pow(x2, 2), torch.pow(x3, 3), torch.pow(x4, 4)], dim=1)


def lqcqp(x):
    # # linear - quadratic - cubic - quartic - penta activation
    h = int(x.shape[1] / 5)
    x1, x2, x3, x4, x5 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h : 3 * h],
        x[:, 3 * h : 4 * h],
        x[:, 4 * h :],
    )
    return torch.cat(
        [x1, torch.pow(x2, 2), torch.pow(x3, 3), torch.pow(x4, 4), torch.pow(x5, 5)],
        dim=1,
    )


def l_e(x):
    # # linear - quadratic - cubic - quartic - penta activation
    h = int(x.shape[1] / 6)
    x1, x2, x3, x4, x5, x6 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h : 3 * h],
        x[:, 3 * h : 4 * h],
        x[:, 4 * h : 5 * h],
        x[:, 5 * h :],
    )
    return torch.cat(
        [
            x1,
            torch.pow(x2, 2),
            torch.pow(x3, 3),
            torch.pow(x4, 4),
            torch.pow(x5, 5),
            torch.pow(x6, 6),
        ],
        dim=1,
    )


def l_s(x):
    # # linear - quadratic - cubic - quartic - penta activation
    h = int(x.shape[1] / 7)
    x1, x2, x3, x4, x5, x6, x7 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h : 3 * h],
        x[:, 3 * h : 4 * h],
        x[:, 4 * h : 5 * h],
        x[:, 5 * h : 6 * h],
        x[:, 6 * h :],
    )
    return torch.cat(
        [
            x1,
            torch.pow(x2, 2),
            torch.pow(x3, 3),
            torch.pow(x4, 4),
            torch.pow(x5, 5),
            torch.pow(x6, 6),
            torch.pow(x7, 7),
        ],
        dim=1,
    )


def l_o(x):
    # # linear - quadratic - cubic - quartic - penta activation
    h = int(x.shape[1] / 8)
    x1, x2, x3, x4, x5, x6, x7, x8 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h : 3 * h],
        x[:, 3 * h : 4 * h],
        x[:, 4 * h : 5 * h],
        x[:, 5 * h : 6 * h],
        x[:, 6 * h : 7 * h],
        x[:, 7 * h :],
    )
    return torch.cat(
        [
            x1,
            torch.pow(x2, 2),
            torch.pow(x3, 3),
            torch.pow(x4, 4),
            torch.pow(x5, 5),
            torch.pow(x6, 6),
            torch.pow(x7, 7),
            torch.pow(x8, 8),
        ],
        dim=1,
    )


def s_d(x):
    h = int(x.shape[1] / 5)
    x1, x2, x3, x4, x5 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h : 3 * h],
        x[:, 3 * h : 4 * h],
        x[:, 4 * h :],
    )
    return torch.cat(
        [
            torch.pow(x1, 2),
            torch.pow(x2, 4),
            torch.pow(x3, 6),
            torch.pow(x4, 8),
            torch.pow(x5, 10),
        ],
        dim=1,
    )


# ReQU: Rectified Quadratic Unit
def requ(x):
    return x * torch.relu(x)


def hyper_tan(x):
    return torch.tanh(x)


def sigm(x):
    return torch.sigmoid(x)


def softplus(x):
    return torch.nn.functional.softplus(x)


def cosh(x):
    return torch.cosh(x) - 1


def rational(x):
    # tanh approximation
    return x / (1 + torch.sqrt(torch.pow(x, 2)))


##################################################################
# DERIVATIVES
##################################################################


def identity_der(x):
    return torch.ones(x.shape)


def step(x):
    sign = torch.sign(x)
    return torch.relu(sign)


def lin_square_der(x):
    h = int(x.shape[1] / 2)
    x1, x2 = x[:h], x[h:]
    return torch.cat([torch.ones(x1.shape), 2 * x2])


def relu_square_der(x):
    h = int(len(x) / 2)
    x1, x2 = x[:h], x[h:]
    return torch.cat([step(x1), 2 * x2])  # torch.pow(x, 2)


def hyper_tan_der(x):
    return torch.ones(x.shape) - torch.pow(torch.tanh(x), 2)


def sigm_der(x):
    y = sigm(x)
    return y * (torch.ones(x.shape) - y)


def softplus_der(x):
    return torch.sigmoid(x)


def sinh(x):
    return torch.sinh(x)


def lqc_der(x):
    # linear - quadratic - cubic derivative
    h = int(x.shape[1] / 3)
    x1, x2, x3 = x[:, :h], x[:, h : 2 * h], x[:, 2 * h :]
    return torch.cat((torch.ones(x1.shape), 2 * x2, 3 * torch.pow(x3, 2)), dim=1)


def lqcq_der(x):
    # # linear - quadratic - cubic - quartic derivative
    h = int(x.shape[1] / 4)
    x1, x2, x3, x4 = x[:, :h], x[:, h : 2 * h], x[:, 2 * h : 3 * h], x[:, 3 * h :]
    return torch.cat(
        (torch.ones(x1.shape), 2 * x2, 3 * torch.pow(x3, 2), 4 * torch.pow(x4, 3)),
        dim=1,
    )


def lqcqp_der(x):
    # # linear - quadratic - cubic - quartic -penta derivative
    h = int(x.shape[1] / 5)
    x1, x2, x3, x4, x5 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h : 3 * h],
        x[:, 3 * h : 4 * h],
        x[:, 4 * h :],
    )
    return torch.cat(
        (
            torch.ones(x1.shape),
            2 * x2,
            3 * torch.pow(x3, 2),
            4 * torch.pow(x4, 3),
            5 * torch.pow(x5, 4),
        ),
        dim=1,
    )


def l_e_der(x):
    # # linear - quadratic - cubic - quartic -penta derivative
    h = int(x.shape[1] / 6)
    x1, x2, x3, x4, x5, x6 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h : 3 * h],
        x[:, 3 * h : 4 * h],
        x[:, 4 * h : 5 * h],
        x[:, 5 * h :],
    )
    return torch.cat(
        (
            torch.ones(x1.shape),
            2 * x2,
            3 * torch.pow(x3, 2),
            4 * torch.pow(x4, 3),
            5 * torch.pow(x5, 4),
            6 * torch.pow(x6, 5),
        ),
        dim=1,
    )


def l_s_der(x):
    # # linear - quadratic - cubic - quartic -penta derivative
    h = int(x.shape[1] / 7)
    x1, x2, x3, x4, x5, x6, x7 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h : 3 * h],
        x[:, 3 * h : 4 * h],
        x[:, 4 * h : 5 * h],
        x[:, 5 * h : 6 * h],
        x[:, 6 * h :],
    )
    return torch.cat(
        (
            torch.ones(x1.shape),
            2 * x2,
            3 * torch.pow(x3, 2),
            4 * torch.pow(x4, 3),
            5 * torch.pow(x5, 4),
            6 * torch.pow(x6, 5),
            7 * torch.pow(x7, 6),
        ),
        dim=1,
    )


def l_o_der(x):
    # # linear - quadratic - cubic - quartic -penta derivative
    h = int(x.shape[1] / 8)
    x1, x2, x3, x4, x5, x6, x7, x8 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h : 3 * h],
        x[:, 3 * h : 4 * h],
        x[:, 4 * h : 5 * h],
        x[:, 5 * h : 6 * h],
        x[:, 6 * h : 7 * h],
        x[:, 7 * h :],
    )
    return torch.cat(
        (
            torch.ones(x1.shape),
            2 * x2,
            3 * torch.pow(x3, 2),
            4 * torch.pow(x4, 3),
            5 * torch.pow(x5, 4),
            6 * torch.pow(x6, 5),
            7 * torch.pow(x7, 6),
            8 * torch.pow(x8, 7),
        ),
        dim=1,
    )


def s_d_der(x):
    h = int(x.shape[1] / 5)
    x1, x2, x3, x4, x5 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h : 3 * h],
        x[:, 3 * h : 4 * h],
        x[:, 4 * h :],
    )
    return torch.cat(
        [
            2 * x1,
            4 * torch.pow(x2, 3),
            6 * torch.pow(x3, 5),
            8 * torch.pow(x4, 7),
            10 * torch.pow(x5, 9),
        ],
        dim=1,
    )


def rational_der(x):
    return 1 / (1 + torch.pow(x, 2))
