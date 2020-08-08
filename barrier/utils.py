import timeit

import numpy as np
import sympy as sp
from z3 import *
import torch
import functools
from shared.activations import activation, activation_der
from shared.activations_symbolic import activation_z3, activation_der_z3


def extract_val_from_z3(model, vars, useSympy):
    """
    :param model: a z3 model
    :param vars: set of vars the model is composed of
    :return: a numpy matrix containing values of the model
    """
    values = []
    for var in vars:
        val = model[var]
        if useSympy:
            values += [to_numpy(val)]
        else:
            values += [RealVal(val)]

    if useSympy:
        return np.matrix(values).T
    else:
        return values


def to_rational(x):
    """
    :param x: a string or numerical representation of a number
    :return: sympy's rational representation
    """
    return sp.Rational(x)


def to_numpy(x):
    """
       :param x: a Z3 numerical representation of a number
       :return: numpy's rational representation
       """
    x = str(x).replace('?', '0')
    return np.float(sp.Rational(x))


def get_symbolic_formula(net, x, xdot):
    """
    :param net:
    :param x:
    :param xdot:
    :return:
    """
    rounding = 2
    z, jacobian = network_until_last_layer(net, x, rounding)

    z = np.dot(np.round(net.layers[-1].weight.data.numpy(), rounding), z)
    # this now contains the gradient \nabla V
    jacobian = np.dot(np.round(net.layers[-1].weight.data.numpy(), rounding), jacobian)

    Vdot = np.dot(jacobian, xdot)
    assert z.shape == (1, 1) and Vdot.shape == (1, 1)
    V = z[0, 0]
    Vdot = Vdot[0, 0]
    # val_in_zero, _ = z3_replacements(z3.simplify(V), V, x, equilibrium)
    # assert z3.simplify(val_in_zero) == 0
    return V, Vdot


def network_until_last_layer(net, x, rounding):
    """
    :param net:
    :param x:
    :param equilibrium:
    :return:
    """
    z = x
    jacobian = np.eye(net.n_inp, net.n_inp)

    for idx, layer in enumerate(net.layers[:-1]):
        w = np.round(layer.weight.data.numpy(), rounding)
        if layer.bias is not None:
            b = np.round(layer.bias.data.numpy(), rounding)[:, None]
        else:
            b = 0
        zhat = np.dot(w, z) + b
        z = activation_z3(net.activation[idx], zhat)
        # Vdot
        jacobian = np.dot(w, jacobian)
        jacobian = np.dot(np.diagflat(activation_der_z3(net.activation[idx], zhat)), jacobian)

    return z, jacobian


def weights_projection(net, equilibrium, rounding, z):
    """
    :param net:
    :param equilibrium:
    :return:
    """
    tol = 10 ** (-rounding)
    # constraints matrix
    c_mat = network_until_last_layer(net, equilibrium, rounding)[0]
    c_mat = sp.Matrix(sp.nsimplify(sp.Matrix(c_mat), rational=True).T)
    # compute projection matrix
    if c_mat == sp.zeros(c_mat.shape[0], c_mat.shape[1]):
        projection_mat = sp.eye(net.layers[-1].weight.shape[1])
    else:
        projection_mat = sp.eye(net.layers[-1].weight.shape[1]) \
                         - c_mat.T * (c_mat @ c_mat.T)**(-1) @ c_mat
    # make the projection w/o gradient operations with torch.no_grad
    last_layer = np.round(net.layers[-1].weight.data.numpy(), rounding)
    last_layer = sp.nsimplify(sp.Matrix(last_layer), rational=True, tolerance=tol)
    new_last_layer = sp.Matrix(last_layer @ projection_mat)

    return new_last_layer


def sympy_replacements(expr, xs, S):
    """
    :param expr: sympy expr
    :param xs: sympy vars
    :param S: list of tensors, batch numerical values
    :return: sum of expr.subs for all elements of S
    """
    total = []
    _is_tensor = isinstance(S, torch.Tensor)
    for idx in range(len(S)):
        if _is_tensor:
            numerical_val = S[idx].data.numpy()
        else:
            numerical_val = S
        replacements = []
        for i in range(len(xs)):
            replacements += [(xs[i, 0], numerical_val[i])]
        total += [expr.subs(replacements)]
    if _is_tensor:
        return torch.tensor(total, dtype=torch.double, requires_grad=True)
    else:
        return total


def z3_replacements(V, Vdot, z3_vars, ctx):
    """
    :param V: z3 expr
    :param Vdot: z3 expr
    :param z3_vars: z3 vars
    :param ctx: list of numerical values
    :return: value of V, Vdot in ctx
    """
    replacements = []
    for i in range(len(z3_vars)):
        replacements += [(z3_vars[i, 0], z3.RealVal(ctx[i, 0]))]
    V_replace = z3.substitute(V, replacements)
    Vdot_replace = z3.substitute(Vdot, replacements)

    return V_replace, Vdot_replace


def print_section(word, k):
    print("=" * 80)
    print(' ', word, ' ', k)
    print("=" * 80)


def compute_equilibria(fx):
    """
    :param fx: list of sympy equations
    :return: list of equilibrium points
    """
    sol = sp.solve(fx)
    return sol


# removes imaginary solutions
def check_real_solutions(sols, x):
    """
    :param sols: list of dictories
    :param x: list of variables
    :return: list of dict w real solutions
    """
    good_sols = []
    for sol in sols:
        is_good_sol = True
        for index in range(len(sol)):
            if sp.im(sol[x[index]]) != 0:
                is_good_sol = False
                break
        if is_good_sol:
            good_sols.append(sol)
    return good_sols


def compute_distance(point, equilibrium):
    """
    :param point: np.array
    :param equilibrium: np.array
    :return: int = squared distance, r^2
    """
    return np.sum(np.power(point - equilibrium, 2))


def compute_bounds(n_vars, f, equilibrium):
    """
    :param n_vars: int, number of variables
    :param f: function
    :param equilibrium: np array
    :return: int, minimum distance from equilibrium to solution points of f
    """
    x0 = equilibrium
    # real=True should consider only real sols
    x_sp = [sp.Symbol('x%d' % i, real=True) for i in range(n_vars)]
    sols = compute_equilibria(f(x_sp))
    # sols = check_real_solutions(sols, x_sp) # removes imaginary solutions
    min_dist = np.inf
    for index in range(len(sols)):
        point = np.array(list(sols[index].values()))  # extract values from dict
        if not (point == x0).all():
            dist = compute_distance(point, x0)
            if dist < min_dist:
                min_dist = dist
    return min_dist


# computes the gradient of V, Vdot in point
# computes a trajectory starting from "point"
# towards increase: + gamma*grad
# towards decrease: - gamma*grad
def compute_trajectory(net, point, f):
    """
    :param net: NN object
    :param point: tensor
    :return: list of tensors
    """
    # set some parameters
    gamma = 0.01  # step-size factor
    max_iters = 20
    # fixing possible dimensionality issues
    trajectory = [point]
    num_vdot_value_old = 0
    # gradient computation
    for gradient_loop in range(max_iters):
        # compute gradient of Vdot
        gradient, num_vdot_value = compute_Vdot_grad(net, point, f)
        # set break conditions
        if abs(num_vdot_value_old - num_vdot_value) < 1e-3 or num_vdot_value > 1e6 or (gradient > 1e6).any():
            break
        else:
            num_vdot_value_old = num_vdot_value
        # "detach" and "requires_grad" make the new point "forget" about previous operations
        point = point.clone().detach() + gamma * gradient.clone().detach()
        point.requires_grad = True
        trajectory.append(point)
    # just checking if gradient is numerically unstable
    assert not torch.isnan(torch.stack(trajectory)).any()
    return trajectory


def compute_V_grad(net, point):
    """
    :param net:
    :param point:
    :return:
    """
    num_v = forward_V(net, point)[0]
    num_v.backward()
    grad_v = point.grad
    return grad_v, num_v


def compute_Vdot_grad(net, point, f):
    """
    :param net:
    :param point:
    :return:
    """
    num_v_dot = forward_Vdot(net, point, f)
    num_v_dot.backward()
    grad_v_dot = point.grad
    assert grad_v_dot is not None
    return grad_v_dot, num_v_dot


def forward_V(net, x):
    """
    :param x: tensor of data points
    :param xdot: tensor of data points
    :return:
            V: tensor, evaluation of x in net
    """
    y = x
    for layer in net.layers[:-1]:
        z = layer(y)
        y = activation(z)
    y = torch.matmul(y, net.layers[-1].weight.T)
    return y


def forward_Vdot(net, x, f):
    """
    :param x: tensor of data points
    :param xdot: tensor of data points
    :return:
            Vdot: tensor, evaluation of x in derivative net
    """
    y = x[None, :]
    xdot = torch.stack(f(x))

    jacobian = torch.diag_embed(torch.ones(x.shape[0], net.n_inp))

    for idx, layer in enumerate(net.layers[:-1]):
        z = layer(y)
        y = activation(net.activation[idx], z)
        jacobian = torch.matmul(layer.weight, jacobian)
        jacobian = torch.matmul(torch.diag_embed(activation_der(net.activation[idx], z)), jacobian)

    jacobian = torch.matmul(net.layers[-1].weight, jacobian)

    return torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1)[0]


def timer(t):
    assert isinstance(t, Timer)

    def dec(f):
        @functools.wraps(f)
        def wrapper(*a, **kw):
            t.start()
            x = f(*a, **kw)
            t.stop()
            return x
        return wrapper
    return dec


class Timer:
    def __init__(self):
        self.min = self.max = self.n_updates = self._sum = self._start = 0
        self.reset()

    def reset(self):
        """min diff, in seconds"""
        self.min = 2 ** 63  # arbitrary
        """max diff, in seconds"""
        self.max = 0
        """number of times the timer has been stopped"""
        self.n_updates = 0

        self._sum = 0
        self._start = 0

    def start(self):
        self._start = timeit.default_timer()

    def stop(self):
        now = timeit.default_timer()
        diff = now - self._start
        assert now >= self._start > 0
        self._start = 0
        self.n_updates += 1
        self._sum += diff
        self.min = min(self.min, diff)
        self.max = max(self.max, diff)

    @property
    def avg(self):
        if self.n_updates == 0:
            return 0
        assert self._sum > 0
        return self._sum / self.n_updates

    def __repr__(self):
        return "total={}s,min={}s,max={}s,avg={}s".format(
                self._sum, self.min, self.max, self.avg
        )
