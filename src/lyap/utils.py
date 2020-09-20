from src.shared.utils import *


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
    E, derivative_e = compute_factors(equilibrium, np.matrix(x), lf)

    # gradV = der(NN) * E + dE/dx * NN
    gradV = np.multiply(jacobian, np.broadcast_to(E, jacobian.shape)) \
            + np.multiply(derivative_e, np.broadcast_to(z[0,0], jacobian.shape))
    # Vdot = gradV * f(x)
    Vdot = gradV @ xdot

    if isinstance(E, sp.Add):
        V = sp.expand(z[0, 0] * E)
        Vdot = sp.expand(Vdot[0, 0])
    else:
        V = z[0, 0] * E
        Vdot = Vdot[0, 0]

    return V, Vdot


def compute_factors(equilibrium, x, lf):
    """
    :param equilibrium:
    :param x:
    :param lf: linear factors
    :return:
    """
    if lf == 'linear':
        E, factors, temp = 1, [], []
        for idx in range(equilibrium.shape[0]):
            E *= sp.simplify(sum((x.T - equilibrium[idx, :]).T)[0, 0])
            factors.append(sp.simplify(sum((x.T - equilibrium[idx, :]).T)[0, 0]))
        for idx in range(len(x)):
            temp += [sum(factors)]
        derivative_e = np.vstack(temp).T
    elif lf == 'quadratic':  # quadratic terms
        E, temp = 1, []
        factors = np.full(shape=(equilibrium.shape[0], x.shape[0]), dtype=object, fill_value=0)
        for idx in range(equilibrium.shape[0]): # number of equilibrium points
            E *= sum(np.power((x.T - equilibrium[idx, :].reshape(x.T.shape)), 2).T)[0,0]
            factors[idx] = (x.T - equilibrium[idx, :].reshape(x.T.shape))
        # derivative = 2*(x-eq)*E/E_i
        grad_e = sp.zeros(1, x.shape[0])
        for var in range(x.shape[0]):
            for idx in range(equilibrium.shape[0]):
                grad_e[var] += sp.simplify(
                    E * factors[idx, var] / sum(np.power((x.T - equilibrium[idx, :].reshape(x.T.shape)), 2).T)[0,0]
                            )
        derivative_e = 2 * grad_e
    else:  # no factors
        E, derivative_e = 1.0, 0.0

    return E, derivative_e


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
        z = activation_z3(net.acts[idx], zhat)
        # Vdot
        jacobian = w @ jacobian
        jacobian = np.diagflat(activation_der_z3(net.acts[idx], zhat)) @ jacobian

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
        z = activation_z3(net.acts[idx], zhat)
        # Vdot
        jacobian = w @ jacobian
        jacobian = np.diagflat(activation_der_z3(net.acts[idx], zhat)) @ jacobian

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
    if rounding > 0:
        last_layer = np.round(net.layers[-1].weight.data.numpy(), rounding)
    else:
        last_layer = net.layers[-1].weight.data.numpy()
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
    if isinstance(S, torch.Tensor):
        numerical_val = S.data.numpy()
    else:
        numerical_val = S
    replacements = []
    for i in range(len(xs)):
        replacements += [(xs[i, 0], numerical_val[i])]
    value = expr.subs(replacements)

    return value


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


def forward_V(net, x):
    """
    :param x: tensor of data points
    :param xdot: tensor of data points
    :return:
            V: tensor, evaluation of x in net
    """
    y = x
    for idx, layer in enumerate(net.layers[:-1]):
        z = layer(y)
        y = activation(net.acts[idx], z)
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
    xdot = torch.stack(f(y.T))
    jacobian = torch.diag_embed(torch.ones(x.shape[0], net.input_size))

    for idx, layer in enumerate(net.layers[:-1]):
        z = layer(y)
        y = activation(net.acts[idx], z)
        jacobian = torch.matmul(layer.weight, jacobian)
        jacobian = torch.matmul(torch.diag_embed(activation_der(net.acts[idx], z)), jacobian)

    jacobian = torch.matmul(net.layers[-1].weight, jacobian)

    return torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1)[0]

