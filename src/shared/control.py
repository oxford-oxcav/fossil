import torch
import numpy as np

from src.shared.activations import activation
from src.shared.activations_symbolic import activation_sym


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
        return (torch.norm(Sdot * tau + S, p=2, dim=1) - torch.norm(S, p=2, dim=1)).sum()


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
            loss = 10 * self.loss_enter_unsafe(S, Sdot) + self.loss_der_direction(S, Sdot)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def loss_der_direction(self, S, Sdot):
        return (S @ Sdot.T).diag().sum()
    
    def loss_onestep(self, S, Sdot, tau=0.05):
        return (torch.norm(Sdot * tau + S, p=2, dim=1) - torch.norm(S, p=2, dim=1)).sum()
    
    def loss_enter_unsafe(self, S, Sdot, tau=0.05):
        S_tau = S + tau * Sdot
        return self.XU.check_containment(S_tau).int().sum() 