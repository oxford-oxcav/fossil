# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from numpy import *
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import cm


def plot_benchmark(
    model, domains, certificate=None, levels=[0], xrange=[-3, 3], yrange=[-3, 3]
):

    """Plot the benchmark. If certificate is provided, it is plotted as well.

    Plots the dynamical model phase plane and the domains with coloured labels.
    If a certificate is provided, it plots the levelsets of the certificate, as
    defined by the levels argument.

    Args:
        model (CTModel): Dynamical model of the benchmark.
        sets (dict): label:set pairs of the domains.
        certificate (NNLearner, optional): . Defaults to None.
        levels (list, optional): Level sets of the certificate to plot. Defaults to zero contour.
        xrange (tuple, optional): Range of the x-axis. Defaults to None.
        yrange (tuple, optional): Range of the y-axis. Defaults to None.
    """
    from matplotlib import pyplot as plt

    ax, fig = plt.subplots()
    ax = model.plot(ax, xrange=xrange, yrange=yrange)
    for lab, dom in domains.items():
        dom.plot(fig, ax, label=lab)
    if certificate is not None:
        ax = plot_certificate(certificate, ax=ax, levels=levels)
    ax.legend()
    plt.show()


def plot_certificate(certificate, ax=None, levels=[0]):
    """Plot contours of the certificate.

    Args:
        certificate (NNLearner): certificate to plot.
        ax : matplotlib axis. Defaults to None, in which case an axis is created.
        levels (list, optional): Level sets of the certificate to plot. Defaults to zero contour.
    """
    from matplotlib import pyplot as plt

    ax = ax or plt.gca()
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    XT = torch.tensor(X, dtype=torch.float32)
    YT = torch.tensor(Y, dtype=torch.float32)
    ZT = certificate(torch.cat((XT.reshape(-1, 1), YT.reshape(-1, 1)), dim=1))
    Z = ZT.detach().numpy().reshape(X.shape)
    levels.sort()
    ax.contour(X, Y, Z, levels=levels, colors="black", linestyles="dashed")
    return ax


def plotting_3d(X, Y, B):
    # Plot Barrier functions
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    suf = ax.plot_surface(X, Y, B, rstride=5, cstride=5, alpha=0.5, cmap=cm.jet)
    zero = ax.contour(X, Y, B, 5, zdir="z", offset=0, colors="k")
    # plt.legend(zero.levels, 'B', loc='upper right')

    return ax


def vector_field(f, Xd, Yd, t):
    # Plot phase plane
    gg = np.vstack([Xd.ravel(), Yd.ravel()]).T
    DF = f(torch.tensor(gg).float()).detach().numpy()
    DX = DF[:, 0].reshape(Xd.shape)
    DY = DF[:, 1].reshape(Xd.shape)
    plt.streamplot(
        Xd, Yd, DX, DY, linewidth=0.5, density=0.5, arrowstyle="-|>", arrowsize=1.5
    )


def plot_circle_sets(ax, centre, r, color, legend):
    # plot circular sets
    r = np.sqrt(r)
    theta = np.linspace(0, 2 * np.pi, 50)
    xc = centre[0] + r * cos(theta)
    yc = centre[1] + r * sin(theta)
    ax.plot(xc[:], yc[:], color, linestyle="--", linewidth=2, label=legend)
    plt.legend(loc="upper right")

    return ax


def plot_square_sets(ax, ll_corner, length, color, legend):
    # plot square sets from lower left corner
    p = Rectangle(
        (ll_corner[0], ll_corner[1]),
        length[0],
        length[1],
        linestyle="--",
        color=color,
        linewidth=2,
        fill=False,
        label=legend,
    )
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
    plt.legend(loc="upper right")

    return ax


def plot_parabola(ax, color, legend):
    # plot circular sets
    y = np.linspace(-1.41, 1.41, 100)
    x = -(y**2)
    ax.plot(x[:], y[:], color, linestyle="--", linewidth=1.5, label=legend)
    plt.legend(loc="upper right")

    return ax
