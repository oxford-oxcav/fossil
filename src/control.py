import torch
import numpy as np
import sympy as sp
from scipy import linalg

from src.activations import activation
from src.activations_symbolic import activation_sym
from src.consts import TimeDomain
from src.utils import vprint


class BaseController(torch.nn.Module):
    def __init__(self, dim, layers, activations) -> None:
        super(BaseController, self).__init__()
        self.dim = dim
        self.acts = activations
        self.layers = []

        n_prev = dim
        for k, n_neurons in enumerate(layers):
            layer = torch.nn.Linear(n_prev, n_neurons, bias=False)
            self.register_parameter("W" + str(k), layer.weight)
            # self.register_parameter("b" + str(k), layer.bias)
            self.layers.append(layer)
            n_prev = n_neurons
        output = torch.nn.Linear(n_prev, dim, bias=False)
        self.register_parameter("W" + str(k + 1), output.weight)
        # self.register_parameter("b" + str(k + 1), layer.bias)
        self.layers.append(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for act, layer in zip(self.acts, self.layers):
            z = layer(y)
            y = activation(act, z)
        z = self.layers[-1](y)
        return z

    def to_symbolic(self, x):
        y = x
        rounding = 1
        for act, layer in zip(self.acts, self.layers):
            W = layer.weight.detach().numpy().round(rounding)
            # b = layer.bias.detach().numpy().round(rounding)
            z = np.atleast_2d(W @ y).T  # + b
            y = activation_sym(act, z)
        W = self.layers[-1].weight.detach().numpy().round(rounding)
        # b = self.layers[-1].bias.detach().numpy().round(rounding)
        z = W @ y  # + b

        print(f"Controller: \n{z}")

        return z


class GeneralController(torch.nn.Module):
    """
    GeneralController generalises the Controller module in the dimensionality definition.
    The 'learn' method is empty, because it doesnt "learn" per se, but always in order to
    make the Lyapunov conditions valid.
    Might merge the Base and Generalised Controller in the future...
    """

    def __init__(self, inputs, output, layers, activations) -> None:
        super(GeneralController, self).__init__()
        self.inp = inputs
        self.out = output
        self.acts = activations
        self.layers = []

        n_prev = self.inp
        for k, n_neurons in enumerate(layers):
            layer = torch.nn.Linear(n_prev, n_neurons, bias=False)
            self.register_parameter("W" + str(k), layer.weight)
            # self.register_parameter("b" + str(k), layer.bias)
            self.layers.append(layer)
            n_prev = n_neurons
        output = torch.nn.Linear(n_prev, self.out, bias=False)
        self.register_parameter("W" + str(k + 1), output.weight)
        # self.register_parameter("b" + str(k + 1), layer.bias)
        self.layers.append(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for act, layer in zip(self.acts, self.layers):
            z = layer(y)
            y = activation(act, z)
        z = self.layers[-1](y)
        return z

    def learn(self, x: torch.Tensor, f_open: torch.Tensor, optimizer: torch.optim):
        pass

    def to_symbolic(self, x, verbose=False):
        y = np.atleast_2d(x).T
        rounding = 5

        for act, layer in zip(self.acts, self.layers):
            W = layer.weight.detach().numpy().round(rounding)
            # b = layer.bias.detach().numpy().round(rounding)
            z = np.atleast_2d(W @ y)  # + b
            y = activation_sym(act, z)
        W = self.layers[-1].weight.detach().numpy().round(rounding)
        # b = self.layers[-1].bias.detach().numpy().round(rounding)
        z = W @ y  # + b

        vprint(f"Controller: \n{z}", verbose)

        return z


class StabilityCT(BaseController):
    def learn(self, S: torch.Tensor, f_open, optimizer):
        for i in range(1000):
            Sdot = f_open(S) + self(S)
            # Want Sdot to point towards origin.
            # Reward Sdot for pointing opposite to S
            loss = self.loss_onestep(S, Sdot)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def loss_der_direction(self, S, Sdot):
        return (S @ Sdot.T).diag().sum()

    def loss_onestep(self, S, Sdot, tau=0.05):
        return (
            torch.norm(Sdot * tau + S, p=2, dim=1) - torch.norm(S, p=2, dim=1)
        ).sum()


class StabilityDT(BaseController):
    def learn(self, S: torch.Tensor, f_open, optimizer):
        for i in range(2000):
            Sdot = f_open(S) + self(S)
            # Sdot should be smaller (in norm) than S
            loss = (torch.norm(Sdot, p=2, dim=1) - torch.norm(S, p=2, dim=1)).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


class SafeStableCT(BaseController):
    def __init__(self, dim, layers, activations, unsafe) -> None:
        super().__init__(dim, layers, activations)
        self.XU = unsafe

    def learn(self, S: torch.Tensor, f_open, optimizer):
        for i in range(2000):
            Sdot = f_open(S) + self(S)
            loss = 10 * self.loss_enter_unsafe(S, Sdot) + self.loss_der_direction(
                S, Sdot
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def loss_der_direction(self, S, Sdot):
        return (S @ Sdot.T).diag().sum()

    def loss_onestep(self, S, Sdot, tau=0.05):
        return (
            torch.norm(Sdot * tau + S, p=2, dim=1) - torch.norm(S, p=2, dim=1)
        ).sum()

    def loss_enter_unsafe(self, S, Sdot, tau=0.05):
        S_tau = S + tau * Sdot
        return self.XU.check_containment(S_tau).int().sum()


class TrajectorySafeStableCT(GeneralController):
    """
    TrajectorySafeStable computes the trajectory of a model,
    penalises the entry in the unsafe set and pushes the trajectory in the safe set
    """

    def __init__(
        self, inputs, outputs, layers, activations, time_domain, goal, unsafe, steps=10
    ) -> None:
        super().__init__(inputs, outputs, layers, activations)
        self.XU = unsafe
        self.XG = goal
        self.tau = 0.05
        self.steps = steps
        self.time_dom = time_domain
        if time_domain == TimeDomain.DISCRETE:
            self.tau = 1

    def learn(self, S: torch.Tensor, f_open, optimizer):
        old_loss = 0
        control_epochs = 2000
        for i in range(control_epochs):
            # Sdot = f_open(S) + self(S)
            traj = self.trajectory_compute(f_open, S)
            loss = self.loss_enter_goal(traj) - 0.5 * self.loss_enter_unsafe(traj)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if abs(old_loss - loss.item()) < 1e-6:
                print(f"Learning Converged")
                break
            old_loss = loss.item()
            if i % 200 == 0:
                print(
                    f"Control Loss: {loss.item():.2f}. Progress: {i/control_epochs*100:.0f}%"
                )

    def trajectory_compute(self, f_open, S):
        Sdot = torch.zeros((self.steps, S.shape[0], S.shape[1]))
        for s in range(self.steps):
            # compute f(x)
            ctr = self(S)
            tmp = f_open(S, ctr)
            # x_next = x + tau*f(x)
            nextS = S + self.tau * tmp
            Sdot[s, :, :] = nextS
            # update
            S = tmp
        return Sdot

    def loss_enter_goal(self, traj, lamb=0.7):
        steps = traj.shape[0]
        # weight is a forgetting factor, so trajectory is weighted-sum
        weight = torch.flipud(torch.tensor([lamb**i for i in range(steps)]))
        return (weight @ self.XG.check_containment_grad(traj)).sum()

    def loss_onestep(self, S, Sdot, tau=0.05):
        return (
            torch.norm(Sdot * tau + S, p=2, dim=1) - torch.norm(S, p=2, dim=1)
        ).sum()

    def loss_enter_unsafe(self, traj, lamb=0.7):
        steps = traj.shape[0]
        # weight is a forgetting factor, so trajectory is weighted-sum
        weight = torch.flipud(torch.tensor([lamb**i for i in range(steps)]))
        return (weight @ self.XU.check_containment_grad(traj)).sum()


# Computes the trajectory in order to push them towards the goal set
class TrajectoryStable(GeneralController):
    def __init__(
        self, inputs, outputs, layers, activations, time_domain, equilibrium, steps=10
    ) -> None:
        super().__init__(inputs, outputs, layers, activations)

        self.XG = equilibrium
        self.tau = 0.01  # * np.random.rand(steps)
        # self.tau = np.hstack([np.zeros(1), self.tau])
        self.td = time_domain
        self.steps = steps

    def learn(self, S: torch.Tensor, f_open, optimizer):
        old_loss = 0
        for i in range(2500):
            # Sdot = f_open(S) + self(S)
            traj = self.trajectory_compute(f_open, S)
            loss = self.loss_traj_decrease_enter_goal(traj)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if abs(old_loss - loss.item()) < 1e-9:
                print(f"Learning Converged")
                break
            old_loss = loss.item()
            if i % 200 == 0:
                print(f"Control Learning Loss: {loss.item()}")

    def trajectory_compute(self, f_open, S):
        Sdot = torch.zeros((self.steps, S.shape[0], S.shape[1]))
        for s in range(self.steps):
            # compute f(x)
            ctr = self(S)
            tmp = f_open(S, ctr)
            # x_next = x + tau*f(x)
            if self.td == TimeDomain.CONTINUOUS:
                # euler method
                nextS = S + self.tau * tmp
                # midpoint method:
                # x_next = x + tau * f(x + 0.5 * tau * f(x))
                # timediff = self.tau[s+1]-self.tau[s]
                # tmp2 = f_open(0.5 * timediff * tmp) + self(0.5 * timediff * tmp)
                # nextS = S + timediff * tmp2

            elif self.td == TimeDomain.DISCRETE:
                nextS = tmp

            Sdot[s, :, :] = nextS
            # update
            S = nextS
        return Sdot

    def loss_enter_goal(self, traj, lamb=0.5):
        # weight is a forgetting factor, so trajectory is weighted-sum
        weight = torch.flipud(torch.tensor([lamb**i for i in range(self.steps)]))

        traj_in_equilibrium = (traj - self.XG).norm(2, dim=-1)

        return (weight @ traj_in_equilibrium).sum()

    def loss_traj_decrease_enter_goal(self, traj, lamb=2.0):
        # expdecay defines the desired exponential decay of the trajectory
        # expdecay = (torch.tensor([np.exp(-lamb * self.tau[:i+1].sum()) for i in range(self.steps)]) \
        #                                                                           ).repeat(traj.shape[1], 1)

        expdecay = (
            torch.tensor([np.exp(-lamb * i * self.tau) for i in range(self.steps)])
        ).repeat(traj.shape[1], 1)
        expdecay = expdecay.repeat(traj.shape[-1], 1, 1)

        # traj_in_equilibrium = (traj - self.XG).norm(2, dim=-1)

        init_pos = traj[0, :, :].repeat(self.steps, 1, 1)
        # trajectory as desired norm decay
        desired_trajectory = expdecay.T * init_pos

        return (traj - desired_trajectory).norm(2, dim=-1).sum()


def nonzero_loss1(S: torch.Tensor, Sdot: torch.Tensor):
    # penalise zero control not near the origin
    gamma = torch.norm(S, p=2, dim=1) ** 2
    return (-gamma * torch.norm(Sdot, p=2, dim=1)).mean()


def nonzero_loss1b(S: torch.Tensor, Sdot: torch.Tensor):
    # penalise zero control not near the origin
    # gamma = torch.norm(S, p=2, dim=1) ** 2
    relu = torch.relu
    bias = 0.1
    nonzero_penalty = relu(-torch.norm(Sdot, p=2, dim=1) + bias)
    return (nonzero_penalty).mean()


def nonzero_loss2(S: torch.Tensor, Sdot: torch.Tensor):
    # penalise difference between control in each dimension
    # \sum_i=/=j |u_i - u_j|
    l = 0
    for i in range(S.shape[1] - 1):
        j = i + 1
        l += (torch.abs(Sdot[:, i]) / torch.abs(Sdot[:, j]) - 1).mean()
    return l


def ridge_reg(S: torch.Tensor, Sdot: torch.Tensor):
    return torch.norm(Sdot, p=2, dim=1).mean()


def ridge_reg_param(W: list[torch.Tensor]):
    s = 0
    s = s + torch.mean(torch.stack([torch.norm(Wi, p=2) for Wi in W]))
    return s


def cosine_reg(S: torch.Tensor, Sdot: torch.Tensor):
    l = torch.cosine_similarity(S, Sdot, dim=1)
    l = torch.relu(l) + torch.relu(-l - 0.5)
    return l.mean()


class Lineariser:
    """Linearises the model around the origin, which is assumed to be the equilibrium point"""

    def __init__(self, model):
        self.model = model
        self.n_vars = self.model.n_vars
        self.x = [sp.Symbol("x" + str(i), real=True) for i in range(self.n_vars)]
        # zero = torch.tensor([0.0] * self.n_vars).unsqueeze(0)
        # print(self.model(zero))

    def get_model(self):
        return self.model(self.x)

    def get_jacobian(self):
        J = sp.Matrix(self.model(self.x)).jacobian(self.x)
        return J

    def linearise(self):
        J = self.get_jacobian()
        J_0 = J.subs([(x, 0) for x in self.x])
        return np.array(J_0).astype(np.float64)


class LQR:
    """Generates the LQR controller for a linear model"""

    def __init__(self, A, B=None, Q=None, R=None):
        """Initialises the LQR controller

        Args:
            A (np.ndarray): state matrix of the linear model
            B (np.ndarray, optional): Defaults to Identity.
            Q (np.ndarray, optional): Defaults to Identity.
            R (np.ndarray, optional): Defaults to Identity.
        """
        self.A = A
        self.n_vars = A.shape[0]
        self.B = B if B is not None else np.eye(self.n_vars)
        self.n_u = self.B.shape[1]
        self.Q = Q if Q is not None else np.eye(self.n_vars)
        self.R = R if R is not None else np.eye(self.n_u)

    def solve(self):
        P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K


if __name__ == "__main__":
    from experiments.benchmarks import models

    model = models.NonPoly0()
    l = Lineariser(model)

    A = l.linearise()
    K = LQR(A).solve()
    print(K)
