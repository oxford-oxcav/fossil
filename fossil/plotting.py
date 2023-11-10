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

from fossil import consts


def show():
    plt.show()


def benchmark(
    model, certificate, domains={}, levels=[0], xrange=[-3, 3], yrange=[-3, 3]
):
    """Helper function to plot all salient plots.

    Plots the phase plane, the certificate and its lie derivative.

    Args:
        model (CTModel): Dynamical model of the benchmark.
        sets (dict): label:set pairs of the domains.
        certificate (NNLearner, optional): . Defaults to None.
        levels (list, optional): Level sets of the certificate to plot. Defaults to zero contour.
        xrange (tuple, optional): Range of the x-axis. Defaults to None.
        yrange (tuple, optional): Range of the y-axis. Defaults to None.
    """
    try:
        iter(certificate)
    except TypeError:
        certificate = [certificate]

    levels = [[0, cert.beta] if cert.beta is not None else [0] for cert in certificate]

    ax1 = benchmark_plane(model, certificate, domains, levels, xrange, yrange)

    ax2 = benchmark_3d(certificate, domains, levels, xrange, yrange)

    ax3 = benchmark_lie(model, certificate, domains, levels, xrange, yrange)

    # plotting lie does not work with concurrency. Something to do with autograd call and multiprocessing.
    return (ax1, "plane"), (ax2, "surface")  # , (ax3, "Cdot")


def save_plot_with_tags(ax, config: consts.CegisConfig, plot_type: str):
    prop = config.CERTIFICATE.name
    model = config.SYSTEM
    seed = torch.initial_seed()
    try:
        name = model.__name__
    except AttributeError:
        name = type(model(None).open_loop).__name__

    plot_name = f"model={name}_property={prop}_type={plot_type}_seed={seed}.pdf"
    plot_name = "results/" + plot_name
    try:
        fig = ax.get_figure()
        fig.savefig(plot_name, bbox_inches="tight")
    except AttributeError:
        fig = ax[0].get_figure()
        fig.savefig(plot_name, bbox_inches="tight")
    plt.close(fig)


def benchmark_plane(
    model, certificate=[None], domains={}, levels=[[0]], xrange=[-3, 3], yrange=[-3, 3]
):
    """Plot the benchmark. If certificate is provided, it is plotted as well.

    Plots the dynamical model phase plane and the domains with coloured labels.
    If a certificate is provided, it plots the levelsets of the certificate, as
    defined by the levels argument.

    Args:
        model (CTModel): Dynamical model of the benchmark.
        certificate (NNLearner, optional): . Defaults to None.
        sets (dict): label:set pairs of the domains.
        levels (list, optional): Level sets of the certificate to plot. Defaults to zero contour.
        xrange (tuple, optional): Range of the x-axis. Defaults to None.
        yrange (tuple, optional): Range of the y-axis. Defaults to None.
    """

    fig, ax = plt.subplots()
    ax = model.plot(ax, xrange=xrange, yrange=yrange)
    ax = plot_domains(domains, ax=ax)

    for cert, lev in zip(certificate, levels):
        if cert is not None:
            ax = certificate_countour(cert, ax=ax, levels=lev)

    try:
        xrange = domains[consts.XD].lower_bounds[0], domains[consts.XD].upper_bounds[0]
        yrange = domains[consts.XD].lower_bounds[1], domains[consts.XD].upper_bounds[1]
    except (KeyError, AttributeError):
        pass

    ax = add_legend(ax)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_title("Phase Plane")
    return ax


def benchmark_3d(certificate, domains={}, levels=[[0]], xrange=[-3, 3], yrange=[-3, 3]):
    """Plot surface of the certificate benchmark.  If the domains are provided, they are plotted as well.

    Plots the surface of a 2D learner.
    If a certificate is provided, it plots the levelsets of the certificate, as
    defined by the levels argument.

    Args:
        certificate tuple[NNLearner]:
        sets (dict, option): label:set pairs of the domains.
        levels (list, optional): Level sets of the certificate to plot. Defaults to zero contour.
        xrange (tuple, optional): Range of the x-axis. Defaults to None.
        yrange (tuple, optional): Range of the y-axis. Defaults to None.
    """
    fig = plt.figure()
    axs = []
    l = len(certificate)
    pos = 1

    for cert, lev in zip(certificate, levels):
        ax = fig.add_subplot(1, l, pos, projection="3d")

        ax = certificate_surface(cert, ax=ax, levels=lev, xrange=xrange, yrange=yrange)
        ax = plot_domains(domains, ax)
        ax = add_legend(ax)
        ax.set_title("{} Certificate".format(cert._type))
        axs.append(ax)
        pos += 1
    return axs


def benchmark_lie(
    model, certificate, domains={}, levels=[[0]], xrange=[-3, 3], yrange=[-3, 3]
):
    """Plot the lie derivative of the certificate benchmark.  If the domains are provided, they are plotted as well.
    Args:
    certificate (NNLearner): . Defaults to None.
    sets (dict, option): label:set pairs of the domains.
    levels (list, optional): Level sets of the certificate to plot. Defaults to zero contour.
    xrange (tuple, optional): Range of the x-axis. Defaults to None.
    yrange (tuple, optional): Range of the y-axis. Defaults to None.
    """

    fig = plt.figure()
    axs = []
    l = len(certificate)
    pos = 1
    for cert, lev in zip(certificate, levels):
        ax = fig.add_subplot(1, l, pos, projection="3d")

        ax = certificate_lie(cert, model, ax=ax, xrange=xrange, yrange=yrange)
        ax = plot_domains(domains, ax)
        ax = add_legend(ax)
        ax.set_title("{} Lie Derivative".format(cert._type))
        axs.append(ax)
        pos += 1
    return axs


def plot_domains(domains, ax):
    for lab, dom in domains.items():
        try:
            dom.plot(None, ax, label=lab)
        except AttributeError:
            pass
    return ax


def certificate_surface(
    certificate, ax=None, xrange=[-3, 3], yrange=[-3, 3], levels=[0]
):
    """Plot the surface of the certificate.
    Args:
        certificate (NNLearner): certificate to plot.
        ax : matplotlib axis. Defaults to None, in which case an axis is created.
        levels (list, optional): Level sets of the certificate to plot. Defaults to zero contour.
        xrange (tuple, optional): Range of the x-axis. Defaults to [-3, 3].
        yrange (tuple, optional): Range of the y-axis. Defaults to [-3, 3].
    """
    ax = ax or plt.gca()
    x = np.linspace(xrange[0], xrange[1], 100)
    y = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(x, y)
    XT = torch.tensor(X, dtype=torch.float32)
    YT = torch.tensor(Y, dtype=torch.float32)
    ZT = certificate(torch.cat((XT.reshape(-1, 1), YT.reshape(-1, 1)), dim=1))
    Z = ZT.detach().numpy().reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.7, rstride=5, cstride=5)
    levels.sort()
    ax.contour(
        X,
        Y,
        Z,
        levels=levels,
        colors="k",
        linestyles="dashed",
        linewidths=2.5,
    )
    return ax


def certificate_lie(certificate, model, ax=None, xrange=[-3, 3], yrange=[-3, 3]):
    """Plot the surface of the lie derivative of the certificate."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    x = np.linspace(xrange[0], xrange[1], 100)
    y = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(x, y)
    XT = torch.tensor(X, dtype=torch.float32)
    YT = torch.tensor(Y, dtype=torch.float32)
    ZT = certificate.compute_net_gradnet(
        torch.cat((XT.reshape(-1, 1), YT.reshape(-1, 1)), dim=1)
    )[1]
    Z = ZT.detach().numpy()
    dx, dy = (
        model._f_torch(torch.stack([XT.ravel(), YT.ravel()]).T.float())
        .detach()
        .numpy()
        .T
    )
    df = np.stack([dx, dy], axis=1)
    lie = (df @ Z.T).diagonal()
    lie = lie.reshape(X.shape)
    ax.plot_surface(X, Y, lie, cmap=cm.coolwarm, alpha=0.7, rstride=5, cstride=5)
    ax.contour(
        X,
        Y,
        lie,
        levels=[0],
        colors="k",
        linestyles="dashed",
        linewidths=2.5,
    )
    return ax


def certificate_countour(certificate, ax=None, levels=[0]):
    """Plot contours of the certificate.

    Args:
        certificate (NNLearner): certificate to plot.
        ax : matplotlib axis. Defaults to None, in which case an axis is created.
        levels (list, optional): Level sets of the certificate to plot. Defaults to zero contour.
    """

    ax = ax or plt.gca()
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    XT = torch.tensor(X, dtype=torch.float32)
    YT = torch.tensor(Y, dtype=torch.float32)
    ZT = certificate(torch.cat((XT.reshape(-1, 1), YT.reshape(-1, 1)), dim=1))
    Z = ZT.detach().numpy().reshape(X.shape)
    levels.sort()
    CS = ax.contour(X, Y, Z, levels=levels, colors="black", linestyles="dashed")
    # ax.clabel(CS, inline=True, fontsize=10)
    return ax


def add_legend(ax):
    """Add legend to the axis without duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
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


def grad_cert(certificate, model, ax=None, xrange=[-3, 3], yrange=[-3, 3]):
    """Plot the gradient of the certificate."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    x = np.linspace(xrange[0], xrange[1], 100)
    y = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(x, y)
    XT = torch.tensor(X, dtype=torch.float32)
    YT = torch.tensor(Y, dtype=torch.float32)
    ZT = certificate.compute_net_gradnet(
        torch.cat((XT.reshape(-1, 1), YT.reshape(-1, 1)), dim=1)
    )[1]
    Z = ZT.detach().numpy()

    df = Z  # np.stack([dx, dy], axis=1)

    plt.streamplot(
        X,
        Y,
        df[:, 0].reshape(X.shape),
        df[:, 1].reshape(X.shape),
        linewidth=0.8,
        density=1.5,
        arrowstyle="fancy",
        arrowsize=1.5,
        color="tab:gray",
    )
