# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d

import src.verifier as verifier

inf = 1e300
inf_bounds = [-inf, inf]


def inf_bounds_n(n):
    return [inf_bounds] * n


def get_plot_colour(label):
    if label == "goal":
        return "green", "Goal"
    elif label == "unsafe":
        return "red", "Unsafe"
    elif label == "safe":
        return "tab:cyan", "Safe"
    elif label == "init":
        return "blue", "Initial"
    else:
        return "black", None


def square_init_data(domain, batch_size):
    """
    :param domain: list = [lower_bounds, upper_bounds]
                    lower_bounds, upper_bounds are lists
    :param batch_size: int
    :return:
    """

    r1 = torch.tensor(domain[0])
    r2 = torch.tensor(domain[1])
    square_uniform = (r1 - r2) * torch.rand(batch_size, len(domain[0])) + r2
    return square_uniform


# n-dim generalisation of circle and sphere
def round_init_data(centre, r, batch_size):
    """
    :param centre:
    :param r:
    :param batch_size:
    :return:
    """
    dim = len(centre)
    if dim == 1:
        return segment([centre[0] - r, centre[0] + r], batch_size)
    elif dim == 2:
        return circle_init_data(centre, r, batch_size)
    elif dim == 3:
        return sphere_init_data(centre, r, batch_size)
    else:
        return n_dim_sphere_init_data(centre, r, batch_size)


def slice_nd_init_data(centre, r, batch_size):
    """
    :param centre:
    :param r:
    :param batch_size:
    :return:
    """
    dim = len(centre)
    if dim == 2:
        return slice_init_data(centre, r, batch_size)
    elif dim == 3:
        return slice_3d_init_data(centre, r, batch_size)
    else:
        raise ValueError("Positive orthant not supported for more than 3 dimensions.")


# generates data for x>0, y>0
def slice_init_data(centre, r, batch_size):
    """
    :param centre: list/tuple/tensor containing the 'n' coordinates of the centre
    :param radius: int
    :param batch_size: int
    :return:
    """
    r = np.sqrt(r)
    angle = (np.pi / 2) * torch.rand(batch_size, 1)
    radius = r * torch.rand(batch_size, 1)
    x_coord = radius * np.cos(angle)
    y_coord = radius * np.sin(angle)
    offset = torch.cat([x_coord, y_coord], dim=1)

    return torch.tensor(centre) + offset


# generates data for (X - centre)**2 <= radius
def circle_init_data(centre, r, batch_size):
    """
    :param centre: list/tuple/tensor containing the 'n' coordinates of the centre
    :param radius: int
    :param batch_size: int
    :return:
    """
    border_batch = int(batch_size / 10)
    internal_batch = batch_size - border_batch
    r = np.sqrt(r)
    angle = (2 * np.pi) * torch.rand(internal_batch, 1)
    radius = r * torch.rand(internal_batch, 1)
    x_coord = radius * np.cos(angle)
    y_coord = radius * np.sin(angle)
    offset = torch.cat([x_coord, y_coord], dim=1)

    angle = (2 * np.pi) * torch.rand(border_batch, 1)
    x_coord = r * np.cos(angle)
    y_coord = r * np.sin(angle)
    offset_border = torch.cat([x_coord, y_coord], dim=1)
    offset = torch.cat([offset, offset_border])

    return torch.tensor(centre) + offset


# generates data for (X - centre)**2 <= radius
def sphere_init_data(centre, r, batch_size):
    """
    :param centre: list/tupe/tensor containing the 3 coordinates of the centre
    :param radius: int
    :param batch_size: int
    :return:
    """
    # spherical coordinates
    # x = r sin(theta) cos(phi)
    # y = r sin(theta) sin(phi)
    # z = r cos(theta)
    theta = (2 * np.pi) * torch.rand(batch_size, 1)
    phi = np.pi * torch.rand(batch_size, 1)
    r = np.sqrt(r)
    radius = r * torch.rand(batch_size, 1)
    x_coord = radius * np.sin(theta) * np.cos(phi)
    y_coord = radius * np.sin(theta) * np.sin(phi)
    z_coord = radius * np.cos(theta)
    offset = torch.cat([x_coord, y_coord, z_coord], dim=1)

    return torch.tensor(centre) + offset


# generates points in a n-dim sphere: X**2 <= radius**2
# adapted from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
# method 20: Muller generalised


def n_dim_sphere_init_data(centre, radius, batch_size):

    dim = len(centre)
    u = torch.randn(
        batch_size, dim
    )  # an array of d normally distributed random variables
    norm = torch.sum(u**2, dim=1) ** (0.5)
    r = radius * torch.rand(batch_size, dim) ** (1.0 / dim)
    x = torch.div(r * u, norm[:, None]) + torch.tensor(centre)

    return x


# generates data for (X - centre)**2 <= radius
def slice_3d_init_data(centre, r, batch_size):
    """
    :param centre: list/tupe/tensor containing the 3 coordinates of the centre
    :param radius: int
    :param batch_size: int
    :return:
    """
    # spherical coordinates
    # x = r sin(theta) cos(phi)
    # y = r sin(theta) sin(phi)
    # z = r cos(theta)
    theta = (np.pi / 2) * torch.rand(batch_size, 1)
    phi = np.pi / 2 * torch.rand(batch_size, 1)
    r = np.sqrt(r)
    radius = r * torch.rand(batch_size, 1)
    x_coord = radius * np.sin(theta) * np.cos(phi)
    y_coord = radius * np.sin(theta) * np.sin(phi)
    z_coord = radius * np.cos(theta)
    offset = torch.cat([x_coord, y_coord, z_coord], dim=1)

    return torch.tensor(centre) + offset


def add_corners_2d(domain):
    """
    :param domain: list = [lower_bounds, upper_bounds]
                    lower_bounds, upper_bounds are lists
    :return:
    """

    ll_corner = torch.tensor(domain[0])
    ur_corner = torch.tensor(domain[1])
    lr_corner = torch.tensor([domain[1][0], domain[0][1]])
    ul_corner = torch.tensor([domain[0][0], domain[1][1]])

    return torch.stack([ll_corner, ur_corner, lr_corner, ul_corner])


def segment(dims, batch_size):
    return (dims[1] - dims[0]) * torch.rand(batch_size, 1) + dims[0]


def remove_init_unsafe_from_d(data, initials, unsafes):
    """
    :param data:
    :param initials:
    :param unsafes:
    :return:
    """
    center_init = torch.tensor(initials[0])
    center_unsafe = torch.tensor(unsafes[0])
    new_data = []
    for idx in range(len(data)):
        # if data belongs to init or unsafe, remove it
        if (
            torch.norm(center_init - data[idx]) > initials[1] * 1.2
            and torch.norm(center_unsafe - data[idx]) > unsafes[1] * 1.2
        ):
            new_data.append(data[idx])

    new_data = torch.stack(new_data)

    return new_data


def inf_bounds_n(n):
    inf = 1e300
    inf_bounds = [-inf, inf]
    return [inf_bounds] * n


class Set:
    dreal_functions = verifier.VerifierDReal.solver_fncts()
    z3_functions = verifier.VerifierZ3.solver_fncts()

    def __init__(self) -> None:
        pass

    def generate_domain(self, x):
        raise NotImplementedError

    def generate_data(self, batch_size):
        raise NotImplementedError

    @staticmethod
    def set_functions(x):
        if verifier.VerifierDReal.check_type(x):
            return Set.dreal_functions
        if verifier.VerifierZ3.check_type(x):
            return Set.z3_functions


class Union(Set):
    """
    Set formed by union of S1 and S2
    """

    def __init__(self, S1: Set, S2: Set) -> None:
        self.S1 = S1
        self.S2 = S2

    def generate_domain(self, x):
        f = self.set_functions(x)
        return f["Or"](self.S1.generate_domain(x), self.S2.generate_domain(x))

    def generate_data(self, batch_size):
        X1 = self.S1.generate_data(int(batch_size / 2))
        X2 = self.S2.generate_data(int(batch_size / 2))
        return torch.cat([X1, X2])


class Intersection(Set):
    """
    Set formed by intersection of S1 and S2
    """

    def __init__(self, S1: Set, S2: Set) -> None:
        self.S1 = S1
        self.S2 = S2

    def generate_domain(self, x):
        f = self.set_functions(x)
        return f["And"](self.S1.generate_domain(), self.S2.generate_domain())

    def generate_data(self, batch_size):
        s1 = self.S1.generate_data(batch_size)
        s1 = s1[self.S2.check_containment(s1)]
        s2 = self.S2.generate_data(batch_size)
        s2 = s2[self.S1.check_containment(s2)]
        return torch.cat([s1, s2])


class SetMinus(Set):
    """
    Set formed by S1 \ S2
    """

    def __init__(self, S1: Set, S2: Set) -> None:
        self.S1 = S1
        self.S2 = S2

    def generate_domain(self, x):
        f = self.set_functions(x)
        return f["And"](
            self.S1.generate_domain(x), f["Not"](self.S2.generate_domain(x))
        )

    def generate_data(self, batch_size):
        data = self.S1.generate_data(batch_size)
        data = data[~self.S2.check_containment(data)]
        return data

    def plot(self, *args, **kwargs):
        self.S1.plot(*args, **kwargs)
        self.S2.plot(*args, **kwargs)


class Rectangle(Set):
    def __init__(self, lb: tuple[float, ...], ub: tuple[float, ...], dim_select=None):
        self.name = "square"
        self.lower_bounds = lb
        self.upper_bounds = ub
        self.dimension = len(lb)
        self.dim_select = dim_select

    def generate_domain(self, x):
        """
        param x: data point x
        returns: symbolic formula for domain
        """
        f = self.set_functions(x)
        lower = f["And"](*[self.lower_bounds[i] <= x[i] for i in range(self.dimension)])
        upper = f["And"](*[x[i] <= self.upper_bounds[i] for i in range(self.dimension)])
        return f["And"](lower, upper)

    def generate_boundary(self, x):
        """Returns boundary of the rectangle

        Args:
            x (List): symbolic data point

        Returns:
            symbolic formula for boundary of the rectangle
        """

        f = self.set_functions(x)
        lower = f["And"](*[self.lower_bounds[i] == x[i] for i in range(self.dimension)])
        upper = f["And"](*[x[i] == self.upper_bounds[i] for i in range(self.dimension)])
        return f["And"](lower, upper)

    def generate_interior(self, x):
        """Returns interior of the rectangle

        Args:
            x (List): symbolic data point
        """
        f = self.set_functions(x)
        lower = f["And"](*[self.lower_bounds[i] < x[i] for i in range(self.dimension)])
        upper = f["And"](*[x[i] < self.upper_bounds[i] for i in range(self.dimension)])
        return f["And"](lower, upper)

    def generate_complement(self, x):
        """Generates complement of the set as a symbolic formulas

        Args:
            x (list): symbolic data point

        Returns:
            SMT variable: symbolic representation of complement of the rectangle
        """
        f = self.set_functions(x)
        return f["Not"](self.generate_domain(x))

    def generate_data(self, batch_size):
        """
        param x: data point x
        returns: data points generated in relevant domain according to shape
        """
        return square_init_data([self.lower_bounds, self.upper_bounds], batch_size)

    def check_containment(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim_select:
            x = [x[:, i] for i in self.dim_select]
        all_constr = torch.logical_and(
            torch.tensor(self.upper_bounds) >= x, torch.tensor(self.lower_bounds) <= x
        )
        ans = torch.zeros((x.shape[0]))
        for idx in range(all_constr.shape[0]):
            ans[idx] = all_constr[idx, :].all()

        return ans.bool()

    def check_containment_grad(self, x: torch.Tensor) -> torch.Tensor:
        # check containment and return a tensor with gradient
        if self.dim_select:
            x = [x[:, i] for i in self.dim_select]

        # returns 0 if it IS contained, a positive number otherwise
        return torch.relu(
            torch.sum(x - torch.tensor(self.upper_bounds), dim=1)
        ) + torch.relu(torch.sum(torch.tensor(self.lower_bounds) - x, dim=1))

    def plot(self, fig, ax, label=None):
        """
        Plots the set
        """
        if self.dimension != 2:
            raise NotImplementedError("Plotting is only implemented for 2D sets")
        anchor = (self.lower_bounds[0], self.lower_bounds[1])
        width = self.upper_bounds[0] - self.lower_bounds[0]
        height = self.upper_bounds[1] - self.lower_bounds[1]
        colour, label = get_plot_colour(label)
        rect = plt.Rectangle(
            anchor, width, height, fill=False, color=colour, label=label, linewidth=2.5
        )
        ax.add_artist(rect)

        if ax.name == "3d":
            art3d.pathpatch_2d_to_3d(rect, z=0, zdir="z")
        return fig, ax


class Sphere(Set):
    def __init__(self, centre, radius, dim_select=None):
        self.centre = centre
        self.radius = radius
        self.dimension = len(centre)
        self.dim_select = dim_select

    def generate_domain(self, x):
        """
        param x: data point x
        returns: symbolic formula for domain
        """
        if self.dim_select:
            x = [x[i] for i in self.dim_select]
        f = self.set_functions(x)
        return f["And"](
            sum([(x[i] - self.centre[i]) ** 2 for i in range(len(x))])
            <= self.radius**2
        )

    def generate_boundary(self, x):
        """
        param x: data point x
        returns: symbolic formula for domain boundary
        """
        f = self.set_functions(x)
        if self.dim_select:
            x = [x[i] for i in self.dim_select]
        return f["And"](
            sum([(x[i] - self.centre[i]) ** 2 for i in range(self.dimension)])
            == self.radius**2
        )

    def generate_interior(self, x):
        """Returns interior of the sphere

        Args:
            x (List): symbolic data point x

        Returns:
            symbolic formula for interior of the sphere
        """
        f = self.set_functions(x)
        if self.dim_select:
            x = [x[i] for i in self.dim_select]
        return f["And"](
            sum([(x[i] - self.centre[i]) ** 2 for i in range(self.dimension)])
            < self.radius**2
        )

    def generate_complement(self, x):
        """Generates complement of the set as a symbolic formulas

        Args:
            x (list): symbolic data point

        Returns:
            SMT variable: symbolic representation of complement of the sphere
        """
        f = self.set_functions(x)
        return f["Not"](self.generate_domain(x))

    def generate_data(self, batch_size):
        """
        param batch_size: number of data points to generate
        returns: data points generated in relevant domain according to shape
        """
        return round_init_data(self.centre, self.radius**2, batch_size)

    def check_containment(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim_select:
            x = [x[:, i] for i in self.dim_select]
        c = torch.tensor(self.centre).reshape(1, -1)
        return (x - c).norm(2, dim=-1) <= self.radius

    def check_containment_grad(self, x: torch.Tensor) -> torch.Tensor:
        # check containment and return a tensor with gradient
        c = torch.tensor(self.centre).reshape(1, -1)
        if self.dim_select:
            x = x[:, :, self.dim_select]
            c = [self.centre[i] for i in self.dim_select]
            c = torch.tensor(c).reshape(1, -1)
        # returns 0 if it IS contained, a positive number otherwise
        return torch.relu((x - c).norm(2, dim=-1) - self.radius)

    def plot(self, fig, ax, label=None):
        if self.dimension != 2:
            raise NotImplementedError("Plotting only supported for 2D sets")
        colour, label = get_plot_colour(label)
        r = self.radius
        theta = np.linspace(0, 2 * np.pi, 50)
        xc = self.centre[0] + r * np.cos(theta)
        yc = self.centre[1] + r * np.sin(theta)
        ax.plot(xc[:], yc[:], colour, linewidth=2, label=label)
        return fig, ax
        # circle = plt.Circle(
        #     self.centre,
        #     self.radius,
        #     color=colour,
        #     fill=False,
        #     label=label,
        #     linewidth=2.5,
        # )
        # ax.add_patch(circle)
        # if ax.name == "3d":
        #     art3d.pathpatch_2d_to_3d(circle, z=0, zdir="z")
        # return fig, ax


class PositiveOrthantSphere(Set):
    def __init__(self, centre, radius):
        self.name = "positive_sphere"
        self.centre = centre
        self.radius = radius
        self.dimension = len(centre)

    def generate_domain(self, x):
        """
        param x: data point x
        param _And: And function for verifier
        returns: symbolic formula for domain
        """
        fcns = self.set_functions(x)
        _And = fcns["And"]
        return _And(
            *[x_i > 0 for x_i in x],
            sum([(x[i] - self.centre[i]) ** 2 for i in range(self.dimension)])
            <= self.radius**2
        )

    def generate_data(self, batch_size):
        """
        param batch_size: number of data points to generate
        returns: data points generated in relevant domain according to shape
        """
        return slice_nd_init_data(self.centre, self.radius**2, batch_size)


class Torus(Set):
    """
    Torus-shaped set characterised as a sphere of radius outer_radius \setminus a sphere of radius inner_radius
    """

    def __init__(self, centre, outer_radius, inner_radius, dim_select=None):
        self.centre = centre
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.dimension = len(centre)
        self.dim_select = dim_select

        assert outer_radius > inner_radius

    def generate_domain(self, x):
        """
        param x: data point x
        returns: symbolic formula for domain of hyper torus
        """
        if self.dim_select:
            x = [x[i] for i in self.dim_select]
        f = self.set_functions(x)
        return f["And"](
            self.inner_radius**2
            <= sum([(x[i] - self.centre[i]) ** 2 for i in range(len(x))]),
            sum([(x[i] - self.centre[i]) ** 2 for i in range(len(x))])
            <= self.outer_radius**2,
        )

    def generate_boundary(self, x):
        """
        param x: data point x
        returns: symbolic formula for domain boundary
        """
        f = self.set_functions(x)
        return f["Or"](
            (
                sum([(x[i] - self.centre[i]) ** 2 for i in range(self.dimension)])
                == self.inner_radius**2
            ),
            sum([(x[i] - self.centre[i]) ** 2 for i in range(self.dimension)])
            == self.outer_radius**2,
        )

    def generate_data(self, batch_size):
        """
        param batch_size: number of data points to generate
        returns: data points generated in relevant domain according to shape
        """
        return round_init_data(self.centre, self.outer_radius**2, batch_size)

    def check_containment(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim_select:
            x = [x[:, i] for i in self.dim_select]
        c = torch.tensor(self.centre).reshape(1, -1)
        return torch.logical_and(
            (self.inner_radius <= (x - c).norm(2, dim=1)),
            (x - c).norm(2, dim=1) <= self.outer_radius,
        )

    def check_containment_grad(self, x: torch.Tensor) -> torch.Tensor:
        # check containment and return a tensor with gradient
        c = torch.tensor(self.centre).reshape(1, -1)
        if self.dim_select:
            x = x[:, :, self.dim_select]
            c = [self.centre[i] for i in self.dim_select]
            c = torch.tensor(c).reshape(1, -1)

        # returns 0 if it IS contained, a positive number otherwise
        return torch.relu(self.inner_radius - (x - c).norm(2, dim=-1)) + torch.relu(
            (x - c).norm(2, dim=-1) - self.outer_radius
        )

    def plot(self, fig, ax, label=None):
        if self.dimension != 2:
            raise NotImplementedError("Plotting only supported for 2D sets")
        colour, label = get_plot_colour(label)
        r = self.inner_radius
        theta = np.linspace(0, 2 * np.pi, 50)
        xc = self.centre[0] + r * np.cos(theta)
        yc = self.centre[1] + r * np.sin(theta)
        ax.plot(xc[:], yc[:], colour, linewidth=2, label=label)

        r = self.outer_radius
        xc = self.centre[0] + r * np.cos(theta)
        yc = self.centre[1] + r * np.sin(theta)
        ax.plot(xc[:], yc[:], colour, linewidth=2, label=label)

        return fig, ax


class Bean2D(Set):
    """2D Bean shaped (nonconvex) set."""

    def __init__(self, centre: list, radius: float):
        assert len(centre) == 2
        self.centre = centre
        self.radius = radius
        self.dimension = len(centre)

    def generate_domain(self, x):
        """
        param x: data point x
        returns: symbolic formula for domain of hyper torus
        """
        x, y = x
        x = x - self.centre[0]
        y = y - self.centre[1]
        boundary = (x**2 + y**2) ** 2 - self.radius * (x**3 - y**3)
        return boundary <= 0

    def generate_boundary(self, x):
        """
        param x: data point x
        returns: symbolic formula for domain boundary
        """
        x, y = x
        x = x - self.centre[0]
        y = y - self.centre[1]
        boundary = (x**2 + y**2) ** 2 - self.radius * (x**3 - y**3)
        return boundary == 0

    def generate_data(self, batch_size):
        """
        param batch_size: number of data points to generate
        returns: data points generated in relevant domain according to shape
        """
        raise NotImplementedError

    def plot(self, fig, ax):
        """
        Plot the set
        """
        x = np.linspace(-10, 10, 500)
        y = np.linspace(-10, 10, 500)
        X, Y = np.meshgrid(x, y)
        xs = self.centre[0]
        ys = self.centre[1]
        Z = ((X - xs) ** 2 + (Y - ys) ** 2) ** 2 - self.radius * (
            (X - xs) ** 3 + (Y - ys) ** 3
        )
        ax.contour(X, Y, Z, levels=[0], colors="r")
        return fig, ax


class EmptySet(Set):
    """Empty Set. Generates domains s.t. negations are always unsat."""

    def generate_domain(self, x):
        f = self.set_functions(x)
        return f["False"]

    def check_containment(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], dtype=torch.bool)

    def generate_data(self, batch_size):
        return None


class Reals(Set):
    """Set of R^n"""

    # I think this set is useless because its negation is always False
    def __init__(self, dim):
        self.dim = dim

    def generate_domain(self, x):
        f = self.set_functions(x)
        return f["True"]


class Complement(Set):
    """Complement of a set."""

    def __init__(self, set: Set):
        self.set = set

    def generate_domain(self, x):
        f = self.set_functions(x)
        return f["Not"](self.set.generate_domain(x))

    def check_containment(self, x: torch.Tensor) -> torch.Tensor:
        return ~self.set.check_containment(x)

    def check_containment_grad(self, x: torch.Tensor) -> torch.Tensor:
        return -self.set.check_containment_grad(x)

    def generate_boundary(self, x):
        """
        param x: data point x
        returns: symbolic formula for domain boundary
        """
        return self.set.generate_boundary(x)

    def plot(self, *args, **kwargs):
        return self.set.plot(*args, **kwargs)


if __name__ == "__main__":
    A = torch.tensor([[0, 1.0], [1.0, 0], [0, -1.0], [-1.0, 0]])
    c = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    import z3

    x = [z3.Real("x" + str(i)) for i in range(2)]
    P = Bean2D([1, -1], 1)
    fig, ax = plt.subplots()
    # set axis limits
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    P.plot(fig, ax)
    print(P.generate_domain(x))
    print(P.generate_boundary(x))
    plt.show()
