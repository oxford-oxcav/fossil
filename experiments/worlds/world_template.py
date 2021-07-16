from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn

from simbl_common.utils import mat, tensor


class WorldBase(object):
    """WorldBase abstract class.
    """

    def __init__(self, dt=1 / 80, normalization=None, device='cpu'):
        """ Initialization; see `InvertedPendulumWorld`.

        Parameters
        ----------
        dt : float, optional
             The sampling time.
        normalization : tuple, optional
                        A tuple (Tx, Tu) of arrays used to normalize the state and actions. It
                        is so that diag(Tx) *x_norm = x and diag(Tu) * u_norm = u.
        device : string, optional
                 Device for computation.
        """
        super(WorldBase, self).__init__()
        self.device = device
        self.dt = dt

        self.normalization = normalization
        if normalization is not None:
            self.normalization = [tensor(norm).to(device) for norm in self.normalization]
            self.inv_norm = [norm.clone().detach().pow(-1).to(device) for norm in self.normalization]

    def normalize(self, state, action):
        """Normalize states and actions.
        """
        if self.normalization is None:
            return state, action

        Tx_inv, Tu_inv = map(torch.diag, self.inv_norm)
        state = torch.matmul(state, Tx_inv)

        if action is not None:
            action = torch.matmul(action, Tu_inv)

        return state, action

    def denormalize(self, state, action):
        """De-normalize states and actions.
        """
        if self.normalization is None:
            return state, action

        Tx, Tu = map(torch.diag, self.normalization)

        state = torch.matmul(state, Tx)
        if action is not None:
            action = torch.matmul(action, Tu)

        return state, action

    # Optional method
    def linearize(self):
        """Return the linearized system.

        Returns
        -------
        a : ndarray
            The state matrix.
        b : ndarray
            The action matrix.

        """
        return None, None

    def __call__(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        state: ndarray or Tensor
            normalized states [n_batch, n_states].
        action: ndarray or Tensor
            normalized actions [n_batch, n_inputs].

        Returns
        -------
        state_next: Tensor the next state prediction [n_batch, n_states].

        """
        state, action = self.denormalize(state, action)

        state_next = state  # Put your simulation here!
        state_next = self.normalize(state_next, action)[0]

        return state_next

    def closed_loop_sim(self, x_0, K_, n_steps):
        """Simulate the system in a closed loop fasion for input number of steps.

        Parameters
        ----------
        x_0: torch.Tensor
             normalized initial states [n_batch, n_states].
        K_: np.array or torch.nn.Module
            Control law governing the closed loop system simulation
        n_steps: int
                 Number of timesteps to simulate system for.

        Returns
        -------
        x_list: np.array
                Returns the simulated trajectories [n_batch, n_states, n_steps]
        """
        x_ = x_0
        x_list = []
        for s in range(n_steps):
            if isinstance(K_, np.ndarray):
                u = x_ @ tensor(-K_).to(self.device).t()
            elif isinstance(K_, nn.Module):
                u = K_(x_)
            else:
                raise Exception('Controller type not supported!')
            # simulate system with computed input
            x_ = self.__call__(x_, u)
            x_list.append(x_.cpu().view(2).numpy())
        x_list = np.array(x_list)
        return x_list
