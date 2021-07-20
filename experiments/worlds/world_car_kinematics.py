import torch
import numpy as np

from simbl_common.worlds.world_template import WorldBase


class CarKinematicsWorld(WorldBase):
    """A world-class car kinematics model!"""

    def __init__(self, dt=0.01, normalization=None, device='cpu'):
        """Initialize the CarKinematicsWorld object.

        Parameters
        ----------
        dt: float
            Time interval of the simulation.
        normalization: list
            Whether the normalization is enabled or not.
        device: str
            Where will the tensors live ('cpu' or one of the GPUs)
        """
        super(CarKinematicsWorld, self).__init__(dt, normalization, device)

    def __call__(self, state, action):
        """Compute the next state.

        Parameters
        ----------
        state: ndarray or Tensor
            normalized states [n_batch, n_states].
        action: ndarray or Tensor
            normalized actions [n_batch, n_inputs].

        Returns
        -------
        state_next: Tensor
                    The next state prediction [n_batch, n_states].
        """

        # make sure that the instances are tensors.
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        # denormalize input state and actions
        state, action = self.denormalize(state, action)

        # perform foward propogation
        state_dot = torch.stack([action[:, 0] * torch.cos(state[:, 2]),
                                 action[:, 0] * torch.sin(state[:, 2]),
                                 state[:, 1]], dim=1)
        state_next = state + self.dt * state_dot

        # normalize input states and actions
        state_next = self.normalize(state_next, action)[0]

        return state_next
