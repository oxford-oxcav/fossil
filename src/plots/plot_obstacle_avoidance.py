import numpy as np
from numpy import *
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import numpy as np


def plot_implicit(fn, fig, bbox=(-2.0, 2.0)*3):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    # fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 25) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z', colors='k', linewidths=1.0)
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y', colors='k', linewidths=1.0)

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x', colors='k', linewidths=1.0)

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

    return ax


def vector_field(Xd, Yd, t):
    # Plot phase plane
    DX, DY = f([Xd, Yd])
    DX = DX / np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
    DY = DY / np.linalg.norm(DY, ord=2, axis=1, keepdims=True)
    plt.streamplot(Xd, Yd, DX, DY, linewidth=0.5,
                   density=0.5, arrowstyle='-|>', arrowsize=1.5)


def plot_circle_sets(centre, r, color, legend):
    # plot circular sets
    theta = np.linspace(0, 2 * np.pi, 50)
    xc = centre[0] + r * cos(theta)
    yc = centre[1] + r * sin(theta)
    ax.plot(xc[:], yc[:], color, linestyle='--', linewidth=2, label=legend)
    plt.legend(loc='upper right')

    return ax


def plot_square_sets(ll_corner, length, color, legend):
    # plot square sets from lower left corner
    p = Rectangle((ll_corner[0], ll_corner[1]), length[0], length[1], linestyle='--', color=color, \
                    linewidth=1.5, fill=False, label=legend)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
    plt.legend(loc='upper right')

    return ax


# from https://stackoverflow.com/questions/44881885/python-draw-parallelepiped
def plot_cube(cube_definition, fig):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='g')
    faces.set_facecolor((0, 1, 0, 0.1))

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

    # ax.set_aspect('equal')
    return ax


def plot_cylinder(x_bounds, r, color, legend):
    # plot circular sets
    # Cylinder
    x = np.linspace(x_bounds[0], x_bounds[1], 1000)
    z = np.linspace(-1, 1, 100)
    Xc, Zc = np.meshgrid(x, z)
    Yc = np.sqrt(r - Xc ** 2)

    ax.plot_surface(Xc, Yc, Zc, color=color)
    ax.plot_surface(Xc, -Yc, Zc, color=color)

    plt.legend(loc='upper right')

    return ax


# obstacle domains:
# X: -2 <= x, y <= 2, -1.57 <= phi, phi <= 1.57
# X_i: -0.1 <= x, x <= 0.1, -2 <= y, y <= -1.8, -0.52 <= phi, phi <= 0.52
# X_u: x**2 + y**2 <= 0.04

def barrier_for_oa(x0,x1,x2):

    return ( - 3.828 * (-2.8280000686645508 - 2.2460000514984131 * x0 + 4.8940000534057617 * x1 - 0.39500001072883606 * x2) + 0.93999999761581421 * pow((-0.28999999165534973 - 1.593000054359436 * x0 - 0.064000003039836884 * x1 + 3.246999979019165 * x2), 2) + 0.67199999094009399 * pow((0.22300000488758087 - 1.3960000276565552 * x0 - 0.88400000333786011 * x1 - 1.2000000476837158 * x2), 3) - 0.69900000095367432 * pow((1 - 0.24199999868869781 * x0 + 1.8760000467300415 * x1 - 0.76700001955032349 * x2), 3) - 1.3500000238418579 * pow((1.1920000314712524 - 0.76200002431869507 * x0 - 1.7070000171661377 * x1 - 0.85000002384185791 * x2), 3))

if __name__ == '__main__':
    fig = plt.figure()
    ax = plot_implicit(barrier_for_oa, fig, bbox=[-2, 2, -2, 2, -1.57, 1.57])

    # unsafe Set
    ax = plot_cylinder([-0.25, 0.25], 0.04, 'r', 'Unsafe Set')

    # Initial Set
    cube_definition = [
                        (-0.1, -2, -0.52), (-0.1, -1.8, -0.52), (0.1, -2, -0.52), (-0.1, -2, 0.52)
                        ]
    ax = plot_cube(cube_definition, fig)


    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$\psi$')
    plt.title('Barrier Certificate')

    ######################
    # plot 2d, with z=0
    #####################

    plt.figure()
    ax = plt.gca()


    def f(v):
        x, y = v
        dydt =[y + 2*x*y, -x - y**2 + 2*x**2]
        return dydt


    # Initial Region
    p = Rectangle((-0.1, -2), 0.2, 0.2, linestyle='--', color='g', \
                  linewidth=1.5, fill=False, label='Initial Set')
    ax.add_patch(p)

    # Unsafe Region
    ax = plot_circle_sets((0.0, 0.0), 0.2, 'r', 'Unsafe Set')


    # plot vector field
    xv = np.linspace(-1.0, 1.0, 100)
    yv = np.linspace(-2.0, 1.0, 100)
    Xv, Yv = np.meshgrid(xv, yv)
    t = np.linspace(0, 5, 100)
    vector_field(Xv, Yv, t)


    X = np.linspace(-1.0, 1.0, 100)
    Y = np.linspace(-2.0, 1.0, 100)
    x1, x2 = np.meshgrid(X, Y)
    B_on_zero_z = barrier_for_oa(x1, x2, 0)

    ax.contour(X, Y, B_on_zero_z, 0, linewidths=2, colors='k')

    plt.title('Barrier Border')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()
