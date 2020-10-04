import numpy as np
from numpy import *
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import cm


def plotting_3d(X, Y, B):
    # Plot Barrier functions
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    suf = ax.plot_surface(X, Y, B, rstride=5, cstride=5, alpha=0.5, cmap=cm.jet)
    zero = ax.contour(X, Y, B, 5, zdir='z', offset=0, colors='k')
    # plt.legend(zero.levels, 'B', loc='upper right')

    return ax


def vector_field(f, Xd, Yd, t):
    # Plot phase plane
    DX, DY = f([Xd, Yd])
    DX = DX / np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
    DY = DY / np.linalg.norm(DY, ord=2, axis=1, keepdims=True)
    plt.streamplot(Xd, Yd, DX, DY, linewidth=0.5,
                   density=0.5, arrowstyle='-|>', arrowsize=1.5)


def plot_circle_sets(ax, centre, r, color, legend):
    # plot circular sets
    r = np.sqrt(r)
    theta = np.linspace(0, 2 * np.pi, 50)
    xc = centre[0] + r * cos(theta)
    yc = centre[1] + r * sin(theta)
    ax.plot(xc[:], yc[:], color, linestyle='--', linewidth=2, label=legend)
    plt.legend(loc='upper right')

    return ax


def plot_square_sets(ax, ll_corner, length, color, legend):
    # plot square sets from lower left corner
    p = Rectangle((ll_corner[0], ll_corner[1]), length[0], length[1], linestyle='--', color=color, \
                    linewidth=2, fill=False, label=legend)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
    plt.legend(loc='upper right')

    return ax


def plot_parabola(ax, color, legend):
    # plot circular sets
    y = np.linspace(-1.41, 1.41, 100)
    x = -y**2
    ax.plot(x[:], y[:], color, linestyle='--', linewidth=1.5, label=legend)
    plt.legend(loc='upper right')

    return ax
