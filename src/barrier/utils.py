from src.shared.utils import *


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
    jacobian = np.eye(net.input_size, net.input_size)

    for idx, layer in enumerate(net.layers[:-1]):
        w = np.round(layer.weight.data.numpy(), rounding)
        if layer.bias is not None:
            b = np.round(layer.bias.data.numpy(), rounding)[:, None]
        else:
            b = 0
        zhat = np.dot(w, z) + b
        z = activation_z3(net.acts[idx], zhat)
        # Vdot
        jacobian = np.dot(w, jacobian)
        jacobian = np.dot(np.diagflat(activation_der_z3(net.acts[idx], zhat)), jacobian)

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


# todo: remove because unused
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


# todo: remove because unused
def forward_Vdot(net, x, f):
    """
    :param x: tensor of data points
    :param xdot: tensor of data points
    :return:
            Vdot: tensor, evaluation of x in derivative net
    """
    y = x[None, :]
    xdot = torch.stack(f(x))

    jacobian = torch.diag_embed(torch.ones(x.shape[0], net.input_size))

    for idx, layer in enumerate(net.layers[:-1]):
        z = layer(y)
        y = activation(net.acts[idx], z)
        jacobian = torch.matmul(layer.weight, jacobian)
        jacobian = torch.matmul(torch.diag_embed(activation_der(net.acts[idx], z)), jacobian)

    jacobian = torch.matmul(net.layers[-1].weight, jacobian)

    return torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1)[0]


