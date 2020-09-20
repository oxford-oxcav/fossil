# pylint: disable=not-callable
import torch
from src.shared.cegis_values import CegisStateKeys
from src.shared.component import Component
from src.shared.consts import LearningFactors
from src.shared.sympy_converter import sympy_converter
from src.shared.utils import *
import sympy as sp
import numpy as np

T = Timer()


class Regulariser(Component):
    def __init__(self, net, x, xdot, eq, rounding):
        super().__init__()
        self.net = net
        self.x = x
        self.xdot = xdot
        self.eq = eq
        self.round = rounding

    @timer(T)
    def get(self, **kw):
        # to disable rounded numbers, set rounding=-1
        sp_handle = kw.get(CegisStateKeys.sp_handle, False)
        fcts = kw[CegisStateKeys.factors]

        if sp_handle:
            f_verifier = kw[CegisStateKeys.verifier_fun]
            x_map = kw[CegisStateKeys.x_v_map]
            x_sp = [sp.Symbol('x%d' % i) for i in range(len(self.x))]
            V_s, Vdot_s = self.get_symbolic_formula(sp.Matrix(x_sp), f_verifier(np.array(x_sp).T), lf=fcts)
            V_s, Vdot_s = sp.simplify(V_s), sp.simplify(Vdot_s)
            V = sympy_converter(x_map, V_s)
            Vdot = sympy_converter(x_map, Vdot_s)
        # verifier handles
        else:
            V, Vdot = self.get_symbolic_formula(self.x, self.xdot.T, lf=fcts)
        # todo: do we want to pass z3.simplify(V) from here or simplification inside verifier?
        # if isinstance(component, Z3Verifier):
        #     V, Vdot = z3.simplify(V), z3.simplify(Vdot)

        return {CegisStateKeys.V: V, CegisStateKeys.V_dot: Vdot}

    def get_symbolic_formula(self, x, xdot, lf=None):
        """
        :param net:
        :param x:
        :param xdot:
        :return:
        """
        # x_sympy = [sp.Symbol('x%d' % i) for i in range(x.shape[0])]
        # x = sp.Matrix(x_sympy)
        sympy_handle = True if isinstance(self.x, sp.Matrix) else False
        if sympy_handle:
            z, jacobian = self.network_until_last_layer_sympy(x)
        else:
            z, jacobian = self.network_until_last_layer(x)

        if self.eq is None:
            equilibrium = np.zeros((self.net.input_size, 1))

        # projected_last_layer = weights_projection(net, equilibrium, rounding, z)
        projected_last_layer = np.round(self.net.layers[-1].weight.data.numpy(), self.round)
        z = projected_last_layer @ z
        # this now contains the gradient \nabla V
        jacobian = projected_last_layer @ jacobian

        assert z.shape == (1, 1)
        # V = NN(x) * E(x)
        E, derivative_e = self.compute_factors(np.matrix(x), lf)

        # gradV = der(NN) * E + dE/dx * NN
        gradV = np.multiply(jacobian, np.broadcast_to(E, jacobian.shape)) \
                + np.multiply(derivative_e, np.broadcast_to(z[0, 0], jacobian.shape))
        # Vdot = gradV * f(x)
        Vdot = gradV @ xdot

        if isinstance(E, sp.Add):
            V = sp.expand(z[0, 0] * E)
            Vdot = sp.expand(Vdot[0, 0])
        else:
            V = z[0, 0] * E
            Vdot = Vdot[0, 0]

        return V, Vdot

    def network_until_last_layer_sympy(self, x):
        """
        :param x:
        :return:
        """
        z = x
        jacobian = np.eye(self.net.input_size, self.net.input_size)

        for idx, layer in enumerate(self.net.layers[:-1]):
            if self.round < 0:
                w = sp.Matrix(layer.weight.data.numpy())
                if layer.bias is not None:
                    b = sp.Matrix(layer.bias.data.numpy()[:, None])
                else:
                    b = sp.zeros(layer.out_features, 1)
            elif self.round > 0:
                w = sp.Matrix(np.round(layer.weight.data.numpy(), self.round))
                if layer.bias is not None:
                    b = sp.Matrix(np.round(layer.bias.data.numpy(), self.round)[:, None])
                else:
                    b = sp.zeros(layer.out_features, 1)

            zhat = w @ z + b
            z = activation_z3(self.net.acts[idx], zhat)
            # Vdot
            jacobian = w @ jacobian
            jacobian = np.diagflat(activation_der_z3(self.net.acts[idx], zhat)) @ jacobian

        return z, jacobian

    def network_until_last_layer(self, x):
        """
        :param x:
        :return:
        """
        z = x
        jacobian = np.eye(self.net.input_size, self.net.input_size)

        for idx, layer in enumerate(self.net.layers[:-1]):
            if self.round < 0:
                w = layer.weight.data.numpy()
                if layer.bias is not None:
                    b = layer.bias.data.numpy()[:, None]
                else:
                    b = np.zeros((layer.out_features, 1))
            elif self.round > 0:
                w = np.round(layer.weight.data.numpy(), self.round)
                if layer.bias is not None:
                    b = np.round(layer.bias.data.numpy(), self.round)[:, None]
                else:
                    b = np.zeros((layer.out_features, 1))

            zhat = w @ z + b
            z = activation_z3(self.net.acts[idx], zhat)
            # Vdot
            jacobian = w @ jacobian
            jacobian = np.diagflat(activation_der_z3(self.net.acts[idx], zhat)) @ jacobian

        return z, jacobian

    def compute_factors(self, x, lf):
        """
        :param x:
        :param lf: linear factors
        :return:
        """
        if lf == LearningFactors.LINEAR:
            E, factors, temp = 1, [], []
            for idx in range(self.eq.shape[0]):
                E *= sp.simplify(sum((x.T - self.eq[idx, :]).T)[0, 0])
                factors.append(sp.simplify(sum((x.T - self.eq[idx, :]).T)[0, 0]))
            for idx in range(len(x)):
                temp += [sum(factors)]
            derivative_e = np.vstack(temp).T
        elif lf == LearningFactors.QUADRATIC:  # quadratic terms
            E, temp = 1, []
            factors = np.full(shape=(self.eq.shape[0], x.shape[0]), dtype=object, fill_value=0)
            for idx in range(self.eq.shape[0]):  # number of equilibrium points
                E *= sum(np.power((x.T - self.eq[idx, :].reshape(x.T.shape)), 2).T)[0, 0]
                factors[idx] = (x.T - self.eq[idx, :].reshape(x.T.shape))
            # derivative = 2*(x-eq)*E/E_i
            grad_e = sp.zeros(1, x.shape[0])
            for var in range(x.shape[0]):
                for idx in range(self.eq.shape[0]):
                    grad_e[var] += sp.simplify(
                        E * factors[idx, var] / sum(np.power((x.T - self.eq[idx, :].reshape(x.T.shape)), 2).T)[0, 0]
                    )
            derivative_e = 2 * grad_e
        else:  # no factors
            E, derivative_e = 1.0, 0.0

        return E, derivative_e

    @staticmethod
    def get_timer():
        return T



