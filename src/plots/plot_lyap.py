# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import numpy as np
import matplotlib.pyplot as plt
from src.plots.plot_fcns import plotting_3d, vector_field
from src.shared.sympy_converter import sympy_converter
import sympy as sp


def set_title_and_label_3d(ax, x_label, y_label, z_label, title):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.title(title)


def plot_lyce(x, V, Vdot, f, title=None):
    plot_limit = 10
    X = np.linspace(-plot_limit, plot_limit, 100)
    Y = np.linspace(-plot_limit, plot_limit, 100)
    x0, x1 = np.meshgrid(X, Y)
    lambda_v = sp.lambdify(x, str(V), modules=['numpy'])
    plot_v = lambda_v([x0], [x1])

    # if no title, assume Lyapunov
    shorthand = 'V'
    longhand = 'Lyapunov'

    if title == 'barrier' or title == 'b':
        shorthand = 'B'
        longhand = 'Barrier'
    elif title == 'lyapunov' or title == 'l':
        shorthand = 'V'
        longhand = 'Lyapunov'
    elif title == 'barrier-lyapunov' or title == 'bl':
        shorthand = 'C'
        longhand = 'Barrier-Lyapunov'

    ax = plotting_3d(x0, x1, plot_v)
    set_title_and_label_3d(ax, '$x$', '$y$', shorthand, longhand + ' function')

    lambda_vdot = sp.lambdify(x, str(Vdot), modules=['numpy'])
    plot_vdot = lambda_vdot([x0], [x1])

    ax = plotting_3d(x0, x1, plot_vdot)
    set_title_and_label_3d(ax, '$x$', '$y$', r'$\dot{%s}$' % shorthand, longhand + ' derivative')
    ################################
    # PLOT 2D -- CONTOUR
    ################################

    plt.figure()
    ax = plt.gca()

    # plot vector field
    xv = np.linspace(-plot_limit, plot_limit, 10)
    yv = np.linspace(-plot_limit, plot_limit, 10)
    Xv, Yv = np.meshgrid(xv, yv)
    t = np.linspace(0, 5, 100)
    vector_field(f, Xv, Yv, t)

    ax.contour(X, Y, plot_v, 5, linewidths=2, colors='k')
    plt.title(longhand + ' Border')
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    plt.show()


def plot_lyapunov_fcn(x, V, f):
    plot_limit = 10
    X = np.linspace(-plot_limit, plot_limit, 100)
    Y = np.linspace(-plot_limit, plot_limit, 100)
    x0, x1 = np.meshgrid(X, Y)
    lambda_f = sp.lambdify(x, str(V), modules=['numpy'])
    lambda_v = sp.lambdify(x, str(V), modules=['numpy'])
    plot_v = lambda_v([x0], [x1])

    ax = plotting_3d(x0, x1, plot_v)
    set_title_and_label_3d(ax, '$x$', '$y$', 'V', 'Lyapunov function')

    ################################
    # PLOT 2D -- CONTOUR
    ################################

    #plt.figure()
    #ax = plt.gca()

    # plot vector field
    #xv = np.linspace(-plot_limit, plot_limit, 10)
    #yv = np.linspace(-plot_limit, plot_limit, 10)
    #Xv, Yv = np.meshgrid(xv, yv)
    #t = np.linspace(0, 5, 100)
    #vector_field(f, Xv, Yv, t)
    #ax.contour(X, Y, plot_v, 5, linewidths=2, colors='k')
    #plt.title('Lyapunov Border')
    #plt.xlabel('$x$')
    #plt.ylabel('$y$')
    plt.show()

def plot_lyce_discrete(x, V, Vdot, f):
    plot_limit = 10
    X = np.linspace(-plot_limit, plot_limit, 100)
    Y = np.linspace(-plot_limit, plot_limit, 100)
    x0, x1 = np.meshgrid(X, Y)
    lambda_v = sp.lambdify(x, str(V), modules=['numpy'])
    plot_v = lambda_v([x0], [x1])

    ax = plotting_3d(x0, x1, plot_v)
    set_title_and_label_3d(ax, '$x$', '$y$', 'V', 'Lyapunov function')

    lambda_vdot = sp.lambdify(x, str(Vdot), modules=['numpy'])
    plot_vdot = lambda_vdot([x0], [x1])

    ax = plotting_3d(x0, x1, plot_vdot)
    set_title_and_label_3d(ax, '$x$', '$y$', '$\dot{V}$', 'Lyapunov derivative')
    ################################
    # PLOT 2D -- CONTOUR
    ################################

    plt.show()