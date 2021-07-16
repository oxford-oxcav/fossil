import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from simbl_common.utils import tensor, saturate
from simbl_common.worlds.world_template import WorldBase


class DoublePendulumWorld(WorldBase):
    """The Double Pendulum world.

    The state is defined as [theta1, theta1_dot, theta2, theta2_dot]
    The input is defined as [input1, input2]
    """

    def __init__(self, mass: list, length: list, center_of_mass: list = None, friction: list = [0, 0], dt=0.01,
                 input_saturation=None, state_saturation=None, normalization=None, device='cpu'):
        """ Double pendulum initialization.

        Parameters
        ----------
        mass: list
            The masses of two links.
        length: list
            The lengths of two links.
        center_of_mass: list
            The center of mass disntances from the hinge.
        friction: list
            The coefficient of friction of the hinges
        dt: float, optional
            The sampling time.
        input_saturation: List[np.array], optional
            The saturation values for input to the actuators.
        state_saturation: List[np.array], optional
            The saturation values for state of the system
        device: str, optional
            The computation device for pyTorch tensors.
        """
        super(DoublePendulumWorld, self).__init__(dt, normalization, device)
        # set masses of links
        self.mass1 = mass[0]
        self.mass2 = mass[1]
        # set lengths of links
        self.length1 = length[0]
        self.length2 = length[1]
        # set center of mass of links
        if center_of_mass is None:
            self.com1 = self.length1 / 2
            self.com2 = self.length2 / 2
        else:
            self.com1 = center_of_mass[0]
            self.com2 = center_of_mass[1]
        # set friction coefficient of links
        self.mu1 = friction[0]
        self.mu2 = friction[1]
        # set the saturation level of input for links
        if input_saturation is None:
            self.input_saturation = [np.asarray([-1, 1]), np.asarray([-1, 1])]
        else:
            self.input_saturation = input_saturation
        # set the saturation level of state for links
        self.state_saturation = state_saturation
        # set constants for the environment
        self.gravity = 9.81
        self.n_inputs = 2
        self.n_states = 4

    @property
    def inertia1(self):
        return self.mass1 * self.length1 ** 2

    @property
    def inertia2(self):
        return self.mass2 * self.length2 ** 2

    def __call__(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        state: ndarray or Tensor
            normalized states [n_natches, n_states].
        action: ndarray or Tensor
            normalized actions [n_batches, n_inputs].

        Returns
        -------
        state_next: Tensor
            The next normalized state of the system

        """
        # saturate input before denormalization
        # action_clipped = saturate(action, self.input_saturation)
        # denormalize the state and action
        state, action = self.denormalize(state, action)
        # get the state derivatives
        state_derivative = self._get_state_derivatives(state, action)
        # perform euler intergration
        state_next = state + self.dt * state_derivative.view(-1, 4)
        # clip state
        # if self.state_saturation is not None:
        #     state_next_clipped = saturate(state_next, self.state_saturation)
        # else:
        #     state_next_clipped = state_next

        # wrap angle from -pi to pi
        state_next[:, 0] = self.wrap(state_next[:, 0], -np.pi, np.pi)
        state_next[:, 2] = self.wrap(state_next[:, 2], -np.pi, np.pi)
        # normalize the next state
        state_next = self.normalize(state_next, action)[0]

        return state_next

    def _get_state_derivatives(self, state, action):
        """ Returns the derivative of the dynamics with respect to state and action.

        Parameters
        ----------
        state: ndarray or Tensor
            denormalized states [n_batches, n_states].
        action: ndarray or Tensor
            denormalized actions [n_batches, n_inputs].

        Returns
        -------
        state_derivatives: Tensor
            State derivatives of the system [n_batches, n_states]
        """
        # split state into constituents
        theta1, theta1_dot, theta2, theta2_dot = torch.chunk(state, self.n_states, dim=1)
        # split action into constituents
        tau1, tau2 = torch.chunk(action, self.n_inputs, dim=1)

        # compute elements of the A matrix
        A_11 = self.inertia1 + \
               self.mass1 * self.com1 ** 2 + \
               self.inertia2 + \
               self.mass2 * self.length1 ** 2 + \
               self.mass2 * self.com2 ** 2 + \
               2 * self.mass2 * self.length1 * self.com2 * torch.cos(theta1)
        A_12 = self.inertia2 + \
               self.mass2 * self.com2 ** 2 + \
               self.mass2 * self.length1 * self.com2 * torch.cos(theta2)
        A_21 = self.inertia2 + \
               self.mass2 * self.com2 ** 2 + \
               self.mass2 * self.length1 * self.com2 * torch.cos(theta2)
        A_22 = tensor(self.inertia2 + self.mass2 * self.com2 ** 2).repeat(A_12.shape)
        # compute elements of the b vector
        b_1 = tau1 - \
              self.mu1 * theta1_dot + \
              self.mass2 * self.gravity * self.com2 * torch.sin(theta1 + theta2) + \
              self.mass2 * self.gravity * self.length1 * torch.sin(theta1) + \
              self.mass1 * self.gravity * self.com1 * torch.sin(theta1) + \
              self.mass2 * self.length1 * self.com2 * theta2_dot ** 2 * torch.sin(theta2) + \
              2 * self.mass2 * self.length1 * self.com2 * theta1_dot * theta2_dot * torch.sin(theta2)
        b_2 = tau2 - \
              self.mu2 * theta2_dot + \
              self.mass2 * self.gravity * self.com2 * torch.sin(theta1 + theta2) - \
              self.mass2 * self.length1 * self.com2 * theta1_dot ** 2 * torch.sin(theta2)

        # concatenate elements to form the A matrix
        A_col_1 = torch.cat((A_11, A_21), dim=1)
        A_col_2 = torch.cat((A_12, A_22), dim=1)
        A = torch.stack((A_col_1, A_col_2), dim=2)
        # concatenate elements to form the B matrix
        b = torch.cat((b_1, b_2), dim=1)

        # compute the second order derivative
        theta_ddot = torch.matmul(torch.inverse(A), b.view(-1, 2, 1))
        theta1_ddot = theta_ddot[:, 0]
        theta2_ddot = theta_ddot[:, 1]
        # concatenate to get the state derivative
        state_derivative = torch.cat([theta1_dot, theta1_ddot, theta2_dot, theta2_ddot], dim=1)

        return state_derivative

    def visualize_frame(self, theta1, theta2, time_index, ax):
        """Make the frame to visualize the environment.

        Parameters
        ----------
        theta1: (np.array) Trajectory angles corresponding to the first link [sim_steps]
        theta2: (np.array) Trajectory angles corresponding to the first link [sim_steps]
        time_index: (int) Time index in the trajectory to plot for
        ax: Axes on which to plot the figure
        """
        # Convert to Cartesian coordinates of the two bob positions.
        x1 = -self.length1 * np.sin(theta1)
        y1 = self.length1 * np.cos(theta1)
        x2 = x1 - self.length2 * np.sin(theta1 + theta2)
        y2 = y1 + self.length2 * np.cos(theta1 + theta2)

        # Plotted bob circle radius
        radius = 0.05
        # Plot a trail of the m2 bob's position for the last trail_secs seconds.
        trail_secs = 1
        # This corresponds to max_trail time points.
        max_trail = int(trail_secs / self.dt)

        # Plot and save an image of the double pendulum configuration for time
        # point i.
        # The pendulum rods.
        ax.plot([0, x1[time_index], x2[time_index]], [0, y1[time_index], y2[time_index]], lw=2, c='k')
        # Circles representing the anchor point of rod 1, and bobs 1 and 2.
        c0 = Circle((0, 0), radius / 2, fc='k', zorder=10)
        c1 = Circle((x1[time_index], y1[time_index]), radius, fc='b', ec='b', zorder=10)
        c2 = Circle((x2[time_index], y2[time_index]), radius, fc='r', ec='r', zorder=10)
        ax.add_patch(c0)
        ax.add_patch(c1)
        ax.add_patch(c2)

        # The trail will be divided into ns segments and plotted as a fading line.
        ns = 20
        s = max_trail // ns

        for j in range(ns):
            imin = time_index - (ns - j) * s
            if imin < 0:
                continue
            imax = imin + s + 1
            # The fading looks better if we square the fractional length along the
            # trail.
            alpha = (j / ns) ** 2
            ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt', lw=2, alpha=alpha)

        # Centre the image on the fixed anchor point, and ensure the axes are equal
        ax.set_xlim(-self.length1 - self.length2 - radius, self.length1 + self.length2 + radius)
        ax.set_ylim(-self.length1 - self.length2 - radius, self.length1 + self.length2 + radius)
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')

    @staticmethod
    def wrap(x, min_value, max_value):
        """Wraps ``x``  around the coordinate system defined by (min_value, max_value).
        For example, min_value = -180, max_value = 180 (degrees), x = 360 --> returns 0.

        :param x: a scalar
        :param min_value: minimum possible value in range
        :param min_value: maximum possible value in range
        """
        diff = max_value - min_value
        while x > max_value:
            x = x - diff
        while x < min_value:
            x = x + diff
        return x
