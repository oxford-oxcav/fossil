# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# definition of various activation fcns

import torch

from fossil import consts


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
    elif select == consts.ActivationType.POLY_2:
        return poly2(p)
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
    elif select == consts.ActivationType.POLY_3:
        return poly3(p)
    elif select == consts.ActivationType.POLY_4:
        return poly4(p)
    elif select == consts.ActivationType.POLY_5:
        return poly5(p)
    elif select == consts.ActivationType.POLY_6:
        return poly6(p)
    elif select == consts.ActivationType.POLY_7:
        return poly_7(p)
    elif select == consts.ActivationType.POLY_8:
        return poly_8(p)
    elif select == consts.ActivationType.EVEN_POLY_4:
        return even_poly4(p)
    elif select == consts.ActivationType.EVEN_POLY_6:
        return even_poly6(p)
    elif select == consts.ActivationType.EVEN_POLY_8:
        return even_poly8(p)
    elif select == consts.ActivationType.EVEN_POLY_10:
        return even_poly10(p)
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
    elif select == consts.ActivationType.POLY_2:
        return poly2_der(p)
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
    elif select == consts.ActivationType.POLY_3:
        return poly3_der(p)
    elif select == consts.ActivationType.POLY_4:
        return poly4_der(p)
    elif select == consts.ActivationType.POLY_5:
        return poly5_der(p)
    elif select == consts.ActivationType.POLY_6:
        return poly_6_der(p)
    elif select == consts.ActivationType.POLY_7:
        return poly7_der(p)
    elif select == consts.ActivationType.POLY_8:
        return poly8_der(p)
    elif select == consts.ActivationType.EVEN_POLY_4:
        return even_poly4_der(p)
    elif select == consts.ActivationType.EVEN_POLY_6:
        return even_poly6_der(p)
    elif select == consts.ActivationType.EVEN_POLY_8:
        return even_poly8_der(p)
    elif select == consts.ActivationType.EVEN_POLY_10:
        return even_poly10_der(p)
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


def poly2(x):
    h = int(x.shape[1] / 2)
    x1, x2 = x[:, :h], x[:, h:]
    return torch.cat([x1, torch.pow(x2, 2)], dim=1)


def relu_square(x):
    h = int(x.shape[1] / 2)
    x1, x2 = x[:, :h], x[:, h:]
    return torch.cat([torch.relu(x1), torch.pow(x2, 2)], dim=1)  # torch.pow(x, 2)


def poly3(x):
    # linear - quadratic - cubic activation
    h = int(x.shape[1] / 3)
    x1, x2, x3 = x[:, :h], x[:, h : 2 * h], x[:, 2 * h :]
    return torch.cat([x1, torch.pow(x2, 2), torch.pow(x3, 3)], dim=1)


def poly4(x):
    # # linear - quadratic - cubic - quartic activation
    h = int(x.shape[1] / 4)
    x1, x2, x3, x4 = x[:, :h], x[:, h : 2 * h], x[:, 2 * h : 3 * h], x[:, 3 * h :]
    return torch.cat([x1, torch.pow(x2, 2), torch.pow(x3, 3), torch.pow(x4, 4)], dim=1)


def poly5(x):
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


def poly6(x):
    # # linear - quadratic - cubic - quartic - penta - sextic activation
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


def poly_7(x):
    # # linear - quadratic - cubic - quartic - penta - sextic - septa activation
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


def poly_8(x):
    # # linear - quadratic - cubic - quartic - penta - sextic - septa - octa activation
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


def even_poly4(x):
    h = int(x.shape[1] / 2)
    x1, x2 = (x[:, :h], x[:, h:])
    return torch.cat(
        [
            torch.pow(x1, 2),
            torch.pow(x2, 4),
        ],
        dim=1,
    )


def even_poly6(x):
    h = int(x.shape[1] / 3)
    x1, x2, x3 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h :],
    )
    return torch.cat(
        [
            torch.pow(x1, 2),
            torch.pow(x2, 4),
            torch.pow(x3, 6),
        ],
        dim=1,
    )


def even_poly8(x):
    h = int(x.shape[1] / 4)
    x1, x2, x3, x4 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h : 3 * h],
        x[:, 3 * h :],
    )
    return torch.cat(
        [
            torch.pow(x1, 2),
            torch.pow(x2, 4),
            torch.pow(x3, 6),
            torch.pow(x4, 8),
        ],
        dim=1,
    )


def even_poly10(x):
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


def poly2_der(x):
    h = int(x.shape[1] / 2)
    x1, x2 = x[:, :h], x[:, h:]
    return torch.cat([torch.ones(x1.shape), 2 * x2], dim=1)


def relu_square_der(x):
    h = int(x.shape[1] / 2)
    x1, x2 = x[:, :h], x[:, h:]
    return torch.cat([step(x1), 2 * x2], dim=1)  # torch.pow(x, 2)


def hyper_tan_der(x):
    return torch.ones(x.shape) - torch.pow(torch.tanh(x), 2)


def sigm_der(x):
    y = sigm(x)
    return y * (torch.ones(x.shape) - y)


def softplus_der(x):
    return torch.sigmoid(x)


def sinh(x):
    return torch.sinh(x)


def poly3_der(x):
    # linear - quadratic - cubic derivative
    h = int(x.shape[1] / 3)
    x1, x2, x3 = x[:, :h], x[:, h : 2 * h], x[:, 2 * h :]
    return torch.cat((torch.ones(x1.shape), 2 * x2, 3 * torch.pow(x3, 2)), dim=1)


def poly4_der(x):
    # # linear - quadratic - cubic - quartic derivative
    h = int(x.shape[1] / 4)
    x1, x2, x3, x4 = x[:, :h], x[:, h : 2 * h], x[:, 2 * h : 3 * h], x[:, 3 * h :]
    return torch.cat(
        (torch.ones(x1.shape), 2 * x2, 3 * torch.pow(x3, 2), 4 * torch.pow(x4, 3)),
        dim=1,
    )


def poly5_der(x):
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


def poly_6_der(x):
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


def poly7_der(x):
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


def poly8_der(x):
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


def even_poly4_der(x):
    h = int(x.shape[1] / 2)
    x1, x2 = (
        x[:, :h],
        x[:, h:],
    )
    return torch.cat(
        [
            2 * x1,
            4 * torch.pow(x2, 3),
        ],
        dim=1,
    )


def even_poly6_der(x):
    h = int(x.shape[1] / 3)
    x1, x2, x3 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h :],
    )
    return torch.cat(
        [
            2 * x1,
            4 * torch.pow(x2, 3),
            6 * torch.pow(x3, 5),
        ],
        dim=1,
    )


def even_poly8_der(x):
    h = int(x.shape[1] / 4)
    x1, x2, x3, x4 = (
        x[:, :h],
        x[:, h : 2 * h],
        x[:, 2 * h : 3 * h],
        x[:, 3 * h :],
    )
    return torch.cat(
        [
            2 * x1,
            4 * torch.pow(x2, 3),
            6 * torch.pow(x3, 5),
            8 * torch.pow(x4, 7),
        ],
        dim=1,
    )


def even_poly10_der(x):
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
