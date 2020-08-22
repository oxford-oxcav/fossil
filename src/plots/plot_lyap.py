import numpy as np
import matplotlib.pyplot as plt
from src.plots.plot_bc import barrier_3d, plot_square_sets, plot_parabola, vector_field
from matplotlib.patches import Rectangle


if __name__ == '__main__':
    plot_limit = 10
    X = np.linspace(-plot_limit, plot_limit, 100)
    Y = np.linspace(-plot_limit, plot_limit, 100)
    x0, x1 = np.meshgrid(X, Y)
    V = (x0**2 + x1**2)*(0.753*np.maximum(0.0, -549*x0/1000 + 171*x1/250 + 1733/1000) + 0.497*np.maximum(0.0, -51*x0/125 - 1281*x1/1000 + 743/1000) + 0.508*np.maximum(0.0, 39*x0/100 + 643*x1/1000 - 9/20))
    Vdot = (-x0 - x1)*(2*x1*(0.753*np.maximum(0.0, -549*x0/1000 + 171*x1/250 + 1733/1000) + 0.497*np.maximum(0.0, -51*x0/125 - 1281*x1/1000 + 743/1000) + 0.508*np.maximum(0.0, 39*x0/100 + 643*x1/1000 - 9/20)) + (x0**2 + x1**2)*(0.515052*np.heaviside(-549*x0/1000 + 171*x1/250 + 1733/1000, 0) - 0.636657*np.heaviside(-51*x0/125 - 1281*x1/1000 + 743/1000, 0) + 0.326644*np.heaviside(39*x0/100 + 643*x1/1000 - 9/20, 0))) + (-x0**3 + x1)*(2*x0*(0.753*np.maximum(0.0, -549*x0/1000 + 171*x1/250 + 1733/1000) + 0.497*np.maximum(0.0, -51*x0/125 - 1281*x1/1000 + 743/1000) + 0.508*np.maximum(0.0, 39*x0/100 + 643*x1/1000 - 9/20)) + (x0**2 + x1**2)*(-0.413397*np.heaviside(-549*x0/1000 + 171*x1/250 + 1733/1000, 0) - 0.202776*np.heaviside(-51*x0/125 - 1281*x1/1000 + 743/1000, 0) + 0.19812*np.heaviside(39*x0/100 + 643*x1/1000 - 9/20, 0)))

    ax = barrier_3d(x0, x1, V)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('V')
    plt.title('Lyapunov fcn')

    ax = barrier_3d(x0, x1, Vdot)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('Vdot')
    plt.title('Lyapunov derivative')

    ################################
    # PLOT 2D
    ################################

    plt.figure()
    ax = plt.gca()

    def f(v):
        x, y = v
        dydt =[-x**3 + y, -x - y]
        return dydt


    # plot vector field
    xv = np.linspace(-plot_limit, plot_limit, 10)
    yv = np.linspace(-plot_limit, plot_limit, 10)
    Xv, Yv = np.meshgrid(xv, yv)
    t = np.linspace(0, 5, 100)
    vector_field(f, Xv, Yv, t)

    ax.contour(X, Y, V, 5, linewidths=2, colors='k')
    plt.title('Lyapunov Border')
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    plt.figure()
    ax = plt.gca()
    ax.contour(X, Y, Vdot, 5, linewidths=2, colors='k')
    plt.title('Vdot Border')
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    plt.show()
