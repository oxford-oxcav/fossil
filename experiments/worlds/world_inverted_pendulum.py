from __future__ import division, print_function

import numpy as np
import torch
from scipy import signal

from simbl_common.worlds.world_template import WorldBase


class InvertedPendulumWorld(WorldBase):
    """Inverted Pendulum.
    """

    def __init__(self, mass, length, friction=0, dt=1 / 80, normalization=None, device='cpu'):
        """ Initialization of the class.

        Parameters
        ----------
        mass : float
        length : float
        friction : float, optional
        dt : float, optional
            The sampling time.
        normalization : tuple, optional
            A tuple (Tx, Tu) of arrays used to normalize the state and actions. It
            is so that diag(Tx) *x_norm = x and diag(Tu) * u_norm = u.
        """
        super(InvertedPendulumWorld, self).__init__(dt, normalization, device)

        self.mass = mass
        self.length = length
        self.gravity = 9.81
        self.friction = friction

    @property
    def inertia(self):
        """Return inertia of the pendulum."""
        return self.mass * self.length ** 2

    def linearize(self):
        """Return the linearized system.

        Returns
        -------
        a : ndarray
            The state matrix.
        b : ndarray
            The action matrix.

        """

        A = np.array([[0, 1],
                      [self.gravity / self.length, -self.friction / self.inertia]],
                     dtype=np.float32)

        B = np.array([[0],
                      [1 / self.inertia]],
                     dtype=np.float32)

        if self.normalization is not None:
            Tx, Tu = map(torch.diag, self.normalization)
            Tx_inv, Tu_inv = map(torch.diag, self.inv_norm)

            A = np.linalg.multi_dot((Tx_inv.cpu(), A, Tx.cpu()))
            B = np.linalg.multi_dot((Tx_inv.cpu(), B, Tu.cpu()))

        sys = signal.StateSpace(A, B, np.eye(2), np.zeros((2, 1)))
        sysd = sys.to_discrete(self.dt)
        return sysd.A, sysd.B

    def __call__(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        state: ndarray or Tensor
            normalized states.
        action: ndarray or Tensor
            normalized actions.

        Returns
        -------
        x_next: Tensor
            The next normalized state of the system

        """
        state, action = self.denormalize(state, action)
        angle, angular_velocity = torch.chunk(state, 2, dim=1)

        x_ddot = self.gravity / self.length * torch.sin(angle) + action / self.inertia

        if self.friction > 0:
            x_ddot -= self.friction / self.inertia * angular_velocity

        state_derivative = torch.stack([angular_velocity, x_ddot], dim=1)
        # Normalize
        s = state + self.dt * state_derivative.view(-1, 2)
        s = self.normalize(s, action)[0]
        angle_, angular_velocity_ = torch.chunk(s, 2, dim=1)

        state_next = torch.stack([angle_, angular_velocity_], dim=1)
        return state_next.view(-1, 2)
