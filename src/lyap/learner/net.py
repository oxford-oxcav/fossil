import torch
import torch.nn as nn
import numpy as np
import sympy as sp
import z3

from src.lyap.verifier.verifier import Verifier
from src.lyap.verifier.z3verifier import Z3Verifier
from src.shared.cegis_values import CegisStateKeys
from src.shared.consts import LearningFactors
from src.shared.learner import Learner
from src.shared.activations import ActivationType, activation, activation_der
from src.lyap.utils import Timer, timer, get_symbolic_formula
from src.shared.sympy_converter import sympy_converter

T = Timer()


class NN(nn.Module, Learner):
    def __init__(self, input_size, *args, bias=True, activate=ActivationType.LIN_SQUARE, equilibria=0, llo=False):
        super(NN, self).__init__()

        self.input_size = input_size
        n_prev = input_size
        self.eq = equilibria
        self.llo = llo
        self._diagonalise = False
        self.acts = activate
        self._is_there_bias = bias
        self.layers = []
        self.closest_unsat = None
        k = 1
        for n_hid in args:
            layer = nn.Linear(n_prev, n_hid, bias=bias)
            self.register_parameter("W" + str(k), layer.weight)
            if bias:
                self.register_parameter("b" + str(k), layer.bias)
            self.layers.append(layer)
            n_prev = n_hid
            k = k + 1

        # last layer
        layer = nn.Linear(n_prev, 1, bias=False)
        # last layer of ones
        if llo:
            layer.weight = torch.nn.Parameter(torch.ones(layer.weight.shape))
            self.layers.append(layer)
        else:          # free output layer
            self.register_parameter("W" + str(k), layer.weight)
            self.layers.append(layer)

    @staticmethod
    def learner_fncts():
        return {
            'sin': torch.sin,
            'cos': torch.cos,
            'exp': torch.exp,
            'If': lambda cond, _then, _else: _then if cond.item() else _else,
        }

    # generalisation of forward with tensors
    def forward_tensors(self, x, xdot):
        """
        :param x: tensor of data points
        :param xdot: tensor of data points
        :return:
                V: tensor, evaluation of x in net
                Vdot: tensor, evaluation of x in derivative net
                jacobian: tensor, evaluation of grad_net
        """
        y = x
        jacobian = torch.diag_embed(torch.ones(x.shape[0], self.input_size))

        for idx, layer in enumerate(self.layers[:-1]):
            z = layer(y)
            y = activation(self.acts[idx], z)

            jacobian = torch.matmul(layer.weight, jacobian)
            jacobian = torch.matmul(torch.diag_embed(activation_der(self.acts[idx], z)), jacobian)

        numerical_v = torch.matmul(y, self.layers[-1].weight.T)
        jacobian = torch.matmul(self.layers[-1].weight, jacobian)
        numerical_vdot = torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1)

        return numerical_v[:, 0], numerical_vdot, jacobian[:, 0, :]

    def numerical_net(self, S, Sdot, lf):
        assert (len(S) == len(Sdot))

        nn, grad_times_f, grad_nn = self.forward_tensors(S, Sdot)
        # circle = x0*x0 + ... + xN*xN
        circle = torch.pow(S, 2).sum(dim=1)

        E, derivative_e = self.compute_factors(S, lf)

        # define E(x) := (x-eq_0) * ... * (x-eq_N)
        # V = NN(x) * E(x)
        V = nn * E
        # gradV = NN(x) * dE(x)/dx  + der(NN) * E(x)
        # gradV = torch.stack([nn, nn]).T * derivative_e + grad_nn * torch.stack([E, E]).T
        gradV = nn.expand_as(grad_nn.T).T * derivative_e.expand_as(grad_nn) \
                + grad_nn * E.expand_as(grad_nn.T).T
        # Vdot = gradV * f(x)
        Vdot = torch.sum(torch.mul(gradV, Sdot), dim=1)

        return V, Vdot, circle

    def compute_factors(self, S, lf):
        E, factors = 1, []
        with torch.no_grad():
            if lf == LearningFactors.LINEAR:  # linear factors
                for idx in range(self.eq.shape[0]):
                    # S - self.eq == [ x-x_eq, y-y_eq ]
                    # (vector_x - eq_0) = ( x-x_eq + y-y_eq )
                    E *= torch.sum(S - torch.tensor(self.eq[idx, :]), dim=1)
                    factors.append(torch.sum(S - torch.tensor(self.eq[idx, :]), dim=1))
                # E = (x-x_eq0 + y-y_eq0) * (x-x_eq1 + y-y_eq1)
                # dE/dx = (x-x_eq0 + y-y_eq0) + (x-x_eq1 + y-y_eq1) = dE/dy
                grad = []
                for idx in range(self.input_size):
                    grad += [torch.sum(torch.stack(factors), dim=0)]
                derivative_e = torch.stack(grad).T
            elif lf == LearningFactors.QUADRATIC:  # quadratic factors
                # define a tensor to store all the components of the quadratic factors
                # factors[:,:,0] stores [ x-x_eq0, y-y_eq0 ]
                # factors[:,:,1] stores [ x-x_eq1, y-y_eq1 ]
                factors = torch.zeros(S.shape[0], self.input_size, self.eq.shape[0])
                for idx in range(self.eq.shape[0]):
                    # S - self.eq == [ x-x_eq, y-y_eq ]
                    # torch.power(S - self.eq, 2) == [ (x-x_eq)**2, (y-y_eq)**2 ]
                    # (vector_x - eq_0)**2 =  (x-x_eq)**2 + (y-y_eq)**2
                    factors[:, :, idx] = S-torch.tensor(self.eq[idx, :])
                    E *= torch.sum(torch.pow(S-torch.tensor(self.eq[idx, :]), 2), dim=1)

                # derivative = 2*(x-eq)*E/E_i
                grad_e = torch.zeros(S.shape[0], self.input_size)
                for var in range(self.input_size):
                    for idx in range(self.eq.shape[0]):
                        grad_e[:, var] += \
                            E * factors[:, var, idx] / torch.sum(torch.pow(S-torch.tensor(self.eq[idx, :]), 2), dim=1)
                derivative_e = 2*grad_e
            else:
                E, derivative_e = torch.tensor(1.0), torch.tensor(0.0)

        return E, derivative_e
    
    def get(self, **kw):
        return self.learn(kw[CegisStateKeys.optimizer], kw[CegisStateKeys.S],
                          kw[CegisStateKeys.S_dot], kw[CegisStateKeys.factors])

    # backprop algo
    @timer(T)
    def learn(self, optimizer, S, Sdot, factors):
        """
        :param optimizer: torch optimiser
        :param S: tensor of data
        :param Sdot: tensor contain f(data)
        :param factors:
        :return: --
        """

        batch_size = len(S)
        learn_loops = 1000
        margin = 0*0.01

        for t in range(learn_loops):
            optimizer.zero_grad()

            V, Vdot, circle = self.numerical_net(S, Sdot, factors)

            slope = 10 ** (self.order_of_magnitude(max(abs(Vdot)).detach()))
            leaky_relu = torch.nn.LeakyReLU(1 / slope)
            # compute loss function. if last layer of ones (llo), can drop parts with V
            if self.llo:
                learn_accuracy = sum(Vdot <= -margin).item()
                loss = (leaky_relu(Vdot + margin * circle)).mean()
            else:
                learn_accuracy = 0.5 * ( sum(Vdot <= -margin).item() + sum(V >= margin).item() )
                loss = (leaky_relu(Vdot + margin * circle)).mean() + (leaky_relu(-V + margin * circle)).mean()

            if t % 100 == 0 or t == learn_loops-1:
                print(t, "- loss:", loss.item(), "- acc:", learn_accuracy * 100 / batch_size, '%')

            loss.backward()
            optimizer.step()

            if self._diagonalise:
                self.diagonalisation()

            if learn_accuracy == batch_size:
                break

            # if self._is_there_bias:
            #     self.weights_projection()
        return {}

    def diagonalisation(self):
        # makes the weight matrices diagonal. works iff intermediate layers are square matrices
        with torch.no_grad():
            for layer in self.layers[:-1]:
                layer.weight.data = torch.diag(torch.diag(layer.weight))

    def weights_projection(self):
        # bias_vector = self.layers[0].bias
        # constraints matrix
        _, _, c_mat = self.forward_tensors(self.equilibrium, self.equilibrium)
        # compute projection matrix
        if (c_mat == 0).all():
            projection_mat = torch.eye(self.layers[-1].weight.shape[1])
        else:
            projection_mat = torch.eye(self.layers[-1].weight.shape[1]) \
                                  - c_mat.T @ torch.inverse(c_mat @ c_mat.T) @ c_mat
        # make the projection w/o gradient operations with torch.no_grad
        with torch.no_grad():
            self.layers[-1].weight.data = self.layers[-1].weight @ projection_mat
            x0 = torch.zeros((1, self.input_size))
            # v0, _, _ = self.forward_tensors(x0, x0)
            # print('Zero in zero? V(0) = {}'.format(v0.data.item()))

    # todo: mv to utils

    def find_closest_unsat(self, S, Sdot, factors):
        min_dist = float('inf')
        V, Vdot, _ = self.numerical_net(S, Sdot, factors)
        for iii in range(S.shape[0]):
            v = V[iii]
            vdot = Vdot[iii]
            if V[iii].item() < 0 and Vdot[iii].item() > 0:
                dist = S[iii].norm()
                if dist < min_dist:
                    min_dist = dist
        self.closest_unsat = min_dist

    @staticmethod
    def order_of_magnitude(number):
        if number.item() != 0:
            return np.ceil(np.log10(number))
        else:
            return 1.0

    @staticmethod
    def get_timer():
        return T


