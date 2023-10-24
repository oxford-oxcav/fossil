# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Iterable
import timeit
import signal
import functools

import numpy as np
import sympy as sp
from z3 import *
import dreal
import torch

from fossil.activations import activation, activation_der
from fossil.activations_symbolic import activation_sym, activation_der_sym
from fossil.consts import CegisStateKeys, LearningFactors


def check_sympy_expression(state, system):
    V, Vdot = state[CegisStateKeys.V], state[CegisStateKeys.V_dot]
    if z3.is_expr(V):
        f, _, __, _ = system()
        x = [sp.Symbol("x%d" % i, real=True) for i in range(2)]
        xdot = f(x)
        V, Vdot = get_symbolic_formula(
            state[CegisStateKeys.net], sp.Matrix(x), sp.Matrix(xdot)
        )

    return V, Vdot


def get_symbolic_formula(net, x, xdot, equilibrium=None, rounding=3, lf=None):
    """
    :param net:
    :param x:
    :param xdot:
    :return:
    """
    # x_sympy = [sp.Symbol('x%d' % i) for i in range(x.shape[0])]
    # x = sp.Matrix(x_sympy)
    sympy_handle = True if isinstance(x, sp.Matrix) else False
    if sympy_handle:
        z, jacobian = network_until_last_layer_sympy(net, x, rounding)
    else:
        z, jacobian = network_until_last_layer(net, x, rounding)

    if equilibrium is None:
        equilibrium = np.zeros((net.input_size, 1))

    # projected_last_layer = weights_projection(net, equilibrium, rounding, z)
    projected_last_layer = np.round(net.layers[-1].weight.data.numpy(), rounding)
    z = projected_last_layer @ z
    # this now contains the gradient \nabla V
    jacobian = projected_last_layer @ jacobian

    assert z.shape == (1, 1)
    # V = NN(x) * E(x)
    E, derivative_e = compute_factors(equilibrium, np.array(x).reshape(1, -1), lf)

    # gradV = der(NN) * E + dE/dx * NN
    gradV = np.multiply(jacobian, np.broadcast_to(E, jacobian.shape)) + np.multiply(
        derivative_e, np.broadcast_to(z[0, 0], jacobian.shape)
    )
    # Vdot = gradV * f(x)
    Vdot = gradV @ xdot

    if isinstance(E, sp.Add):
        V = sp.expand(z[0, 0] * E)
        Vdot = sp.expand(Vdot[0, 0])
    else:
        V = z[0, 0] * E
        Vdot = Vdot[0, 0]

    return V, Vdot


def network_until_last_layer_sympy(net, x, rounding):
    """
    :param net:
    :param x:
    :param equilibrium:
    :return:
    """
    z = x
    jacobian = np.eye(net.input_size, net.input_size)

    for idx, layer in enumerate(net.layers[:-1]):
        if rounding < 0:
            w = sp.Matrix(layer.weight.data.numpy())
            if layer.bias is not None:
                b = sp.Matrix(layer.bias.data.numpy()[:, None])
            else:
                b = sp.zeros(layer.out_features, 1)
        elif rounding > 0:
            w = sp.Matrix(np.round(layer.weight.data.numpy(), rounding))
            if layer.bias is not None:
                b = sp.Matrix(np.round(layer.bias.data.numpy(), rounding)[:, None])
            else:
                b = sp.zeros(layer.out_features, 1)
        #
        # w = sp.Matrix(sp.nsimplify(w, rational=True))
        # b = sp.Matrix(sp.nsimplify(b, rational=True))

        zhat = w @ z + b
        z = activation_sym(net.acts[idx], zhat)
        # Vdot
        jacobian = w @ jacobian
        jacobian = np.diagflat(activation_der_sym(net.acts[idx], zhat)) @ jacobian

    return z, jacobian


def network_until_last_layer(net, x, rounding):
    """
    :param net:
    :param x:
    :param equilibrium:
    :return:
    """
    z = x
    jacobian = np.eye(net.input_size, net.input_size)

    for idx, layer in enumerate(net.layers[:-1]):
        if rounding < 0:
            w = layer.weight.data.numpy()
            if layer.bias is not None:
                b = layer.bias.data.numpy()[:, None]
            else:
                b = np.zeros((layer.out_features, 1))
        elif rounding > 0:
            w = np.round(layer.weight.data.numpy(), rounding)
            if layer.bias is not None:
                b = np.round(layer.bias.data.numpy(), rounding)[:, None]
            else:
                b = np.zeros((layer.out_features, 1))
        #
        # w = sp.Matrix(sp.nsimplify(w, rational=True))
        # b = sp.Matrix(sp.nsimplify(b, rational=True))

        zhat = w @ z + b
        z = activation_sym(net.acts[idx], zhat)
        # Vdot
        jacobian = w @ jacobian
        jacobian = np.diagflat(activation_der_sym(net.acts[idx], zhat)) @ jacobian

    return z, jacobian


def compute_factors(equilibrium, x, lf):
    """
    :param equilibrium:
    :param x:
    :param lf: linear factors
    :return:
    """
    if lf == LearningFactors.QUADRATIC:  # quadratic terms
        E, temp = 1, []
        factors = np.full(
            shape=(equilibrium.shape[0], x.shape[0]), dtype=object, fill_value=0
        )
        for idx in range(equilibrium.shape[0]):  # number of equilibrium points
            E *= sum(np.power((x.T - equilibrium[idx, :].reshape(x.T.shape)), 2).T)[
                0, 0
            ]
            factors[idx] = x.T - equilibrium[idx, :].reshape(x.T.shape)
        # derivative = 2*(x-eq)*E/E_i
        grad_e = sp.zeros(1, x.shape[0])
        for var in range(x.shape[0]):
            for idx in range(equilibrium.shape[0]):
                grad_e[var] += sp.simplify(
                    E
                    * factors[idx, var]
                    / sum(
                        np.power((x.T - equilibrium[idx, :].reshape(x.T.shape)), 2).T
                    )[0, 0]
                )
        derivative_e = 2 * grad_e
    else:  # no factors
        E, derivative_e = 1.0, 0.0

    return E, derivative_e


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
        projection_mat = (
            sp.eye(net.layers[-1].weight.shape[1])
            - c_mat.T * (c_mat @ c_mat.T) ** (-1) @ c_mat
        )
    # make the projection w/o gradient operations with torch.no_grad
    if rounding > 0:
        last_layer = np.round(net.layers[-1].weight.data.numpy(), rounding)
    else:
        last_layer = net.layers[-1].weight.data.numpy()
    last_layer = sp.nsimplify(sp.Matrix(last_layer), rational=True, tolerance=tol)
    new_last_layer = sp.Matrix(last_layer @ projection_mat)

    return new_last_layer


def z3_replacements(expr, z3_vars, ctx):
    """
    :param expr: z3 expr
    :param z3_vars: z3 vars, matrix
    :param ctx: matrix of numerical values
    :return: value of V, Vdot in ctx
    """
    replacements = []
    for i in range(len(z3_vars)):
        try:
            replacements += [(z3_vars[i, 0], z3.RealVal(ctx[i, 0]))]
        except TypeError:
            replacements += [(z3_vars[i], z3.RealVal(ctx[i, 0]))]

    replaced = z3.substitute(expr, replacements)

    return z3.simplify(replaced)


def dreal_replacements(expr, dr_vars, ctx):
    """
    :param expr: dreal expr
    :param dr_vars: dreal vars, matrix
    :param ctx: matrix of numerical values
    :return: value of V, Vdot in ctx
    """
    try:
        replacements = {dr_vars[i, 0]: float(ctx[i, 0]) for i in range(len(dr_vars))}
    except TypeError:
        replacements = {dr_vars[i]: float(ctx[i, 0]) for i in range(len(dr_vars))}

    return expr.Substitute(replacements)


def z3_to_string(f):
    if len(f.children()) == 0:
        return str(f)
    return str(f.decl()).join(z3_to_string(c) for c in f.children())


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
        return np.array(values).reshape(-1, 1)
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
    x = str(x).replace("?", "0")
    return np.float(sp.Rational(x))


def print_section(word, k):
    print("=" * 80)
    print(" ", word, " ", k)
    print("=" * 80)


def compute_equilibria(fx, x):
    """
    :param fx: list of sympy equations
    :return: list of equilibrium points
    """
    sol = sp.solve(fx, real=True)
    return sol


# removes imaginary solutions
def check_real_solutions(sols, x):
    """
    :param sols: list of dictories
    :param x: list of variables
    :return: list of dict w real solutions
    """
    good_sols = []
    if isinstance(sols, dict):
        sols = [sols]
    for sol in sols:
        is_good_sol = True
        for index in range(len(x)):
            if sp.im(sol[x[index]]) != 0:
                is_good_sol = False
                break
        if is_good_sol:
            good_sols.append(sol)
    return good_sols


def dict_to_array(dict, n):
    """
    :param dict:
    :return:
    """
    array = np.zeros((len(dict), n))
    for idx in range(len(dict)):
        array[idx, :] = list(dict[idx].values())
    return array


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
    x_sp = [sp.Symbol("x%d" % i, real=True) for i in range(n_vars)]
    sols = compute_equilibria(f(x_sp))
    # sols = check_real_solutions(sols, x_sp) # removes imaginary solutions
    min_dist = np.inf
    for index in range(len(sols)):
        try:
            point = np.array(list(sols[index].values()))  # extract values from dict
        except KeyError:
            point = np.array(list(sols.values()))
        if not (point == x0).all():
            dist = compute_distance(point, x0)
            if dist < min_dist:
                min_dist = dist
    return min_dist


# computes the gradient of V, Vdot in point
# computes a 20-step trajectory (20 is arbitrary) starting from point
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
    num_vdot_value_old = -1.0
    # gradient computation
    for gradient_loop in range(max_iters):
        # compute gradient of Vdot
        gradient, num_vdot_value = compute_Vdot_grad(net, point, f)
        # set break conditions
        if (
            abs(num_vdot_value_old - num_vdot_value) < 1e-3
            or num_vdot_value > 1e6
            or (gradient > 1e6).any()
        ):
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


def forward_Vdot(net, x, f):
    """
    :param x: tensor of data points
    :param xdot: tensor of data points
    :return:
            Vdot: tensor, evaluation of x in derivative net
    """
    y = x[None, :]
    xdot = torch.stack(f(y.T))
    jacobian = torch.diag_embed(torch.ones(x.shape[0], net.input_size))

    for idx, layer in enumerate(net.layers[:-1]):
        z = layer(y)
        y = activation(net.acts[idx], z)
        jacobian = torch.matmul(layer.weight, jacobian)
        jacobian = torch.matmul(
            torch.diag_embed(activation_der(net.acts[idx], z)), jacobian
        )

    jacobian = torch.matmul(net.layers[-1].weight, jacobian)

    return torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1)[0]


def vprint(arg, verbose=True):
    if verbose:
        print(*arg)


def rotate(l: list, x: int):
    return l[x:] + l[:x]


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
        self.min = 2**63  # arbitrary
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

    @property
    def sum(self):
        return self._sum

    def __repr__(self):
        return "total={}s,min={}s,max={}s,avg={}s, N={}".format(
            self._sum, self.min, self.max, self.avg, self.n_updates
        )


class Timeout:
    # from https://stackoverflow.com/a/22348885
    # Requires UNIX
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class FailedSynthesis(Exception):
    """Exception raised in Primer if CEGIS fails to synthesise"""

    pass


def is_iterable(x):
    return isinstance(x, Iterable)


def contains_object(x, obj):
    if is_iterable(x):
        return contains_object(next(iter(x)), obj)
    else:
        return isinstance(x, obj)


if __name__ == "__main__":
    import dreal

    x = Real("x")
    xd = dreal.Variable("xd")
    l = np.array([[xd]])
    # l = torch.rand((2,1))
    print(contains_object(l, dreal.Variable))
