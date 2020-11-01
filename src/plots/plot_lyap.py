import numpy as np
import matplotlib.pyplot as plt
from src.plots.plot_fcns import plotting_3d, vector_field
from src.shared.sympy_converter import sympy_converter
from matplotlib.patches import Rectangle
import sympy as sp
import z3 as z3
from src.lyap.utils import get_symbolic_formula


def plot_lyce(x, V, Vdot, f):
    plot_limit = 10
    X = np.linspace(-plot_limit, plot_limit, 100)
    Y = np.linspace(-plot_limit, plot_limit, 100)
    x0, x1 = np.meshgrid(X, Y)
    lambda_v = sp.lambdify(x, str(V), modules=['numpy'])
    plot_v = lambda_v([x0], [x1])

    ax = plotting_3d(x0, x1, plot_v)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('V')
    plt.title('Certificate')

    lambda_vdot = sp.lambdify(x, str(Vdot), modules=['numpy'])
    plot_vdot = lambda_vdot([x0], [x1])

    ax = plotting_3d(x0, x1, plot_vdot)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('Vdot')
    plt.title('Certificate derivative')

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
    plt.title('Lyapunov Border')
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    plt.show()
