import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
from scipy import signal
from random import randint
import scipy
from functools import partial

mat = np.atleast_2d


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.


    x[k+1] = A x[k] + B u[k]

    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # ref Bertsekas, p.151

    # first, try to solve the riccati equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    # compute the LQR gain
    K = np.array(scipy.linalg.inv(B.T @ X @ B + R) * (B.T @ X @ A))
    # K = np.array(K)

    eigVals, eigVecs = scipy.linalg.eig(A - B @ K)

    return K, X, eigVals

def dkf(A, C, Q, R):

    # Solve Riccati equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A.T, C.T, Q, R))

    # Compute the Kalman Gain
    L = np.array(X @ C.T @ np.linalg.inv(R + C @ X @ C.T))

    eigVals, eigVecs = scipy.linalg.eig(A - L @ C)

    return L, X, eigVals


def solve(f, initial_state, input_value, dt=0.001):
    """Solve the model ode"""
    fun = partial(f, input_value=input_value)
    sol = scipy.integrate.solve_ivp(fun, [0, dt], initial_state[0])
    return np.reshape(sol.y[:, -1], [1, sol.y[:, -1].shape[0]])


class AUV_model:
    def __init__(self, Ts=0.5):
        self.name = 'AUV_model'
        self.Ts = Ts
        self.n_inputs = 2
        self.n_outputs = 3
        self.zero_state = mat([0., 0., 0.])
        self.current_state = self.zero_state

    def set_state(self, state):
        self.current_state = state.copy()

    def step(self, inputs):
        output = self.run(self.current_state, inputs, n_steps=1)
        system_output = self.current_state.copy()
        self.current_state = output.copy()
        return self.current_state.copy(), system_output.copy()

    def forward(self, inputs):
        output = self.run(self.current_state, inputs, n_steps=1)
        self.current_state = output.copy()
        return self.current_state.copy(), self.current_state.copy()

    def run(self, initial_state, inputs, n_steps):
        # Input shape [n_steps, 2]
        # Initial state shape [1, 3]
        x = initial_state.copy()
        inputs = np.reshape(inputs, [inputs.shape[0], inputs.shape[1], 1])
        state_list = []
        state_list.append(x.copy())
        for step in range(n_steps):
            B_matrix = np.array([[np.cos(x[0, 2]), 0.],
                            [np.sin(x[0, 2]), 0.],
                            [0., 1.]])
            d_state = (self.Ts * (B_matrix @ inputs[step])).reshape(x.shape)
            x += d_state
            state_list.append(x.copy())
        output = np.reshape(mat(state_list[-1]), [n_steps, x.shape[1]])
        # output = np.zeros([n_steps, self.n_outputs])
        # for step in range(n_steps):
        #     output[step, 0:2] = output_angle[step, 0:2]
        #     output[step, 2] = np.sin(output_angle[step, 2])
        #     output[step, 3] = np.cos(output_angle[step, 2])
        return output

    def generate_data(self, batch_size, n_steps):
        # n_input_steps = int(n_steps/(10/self.Ts))  # one step every 1000 s
        # np.random.seed(0)
        # random_steps = (np.random.rand(batch_size, n_input_steps, self.n_inputs) - 0.5)*2
        # np.random.seed(1)
        # random_steps_test = (np.random.rand(batch_size, n_input_steps, self.n_inputs) - 0.5)*2
        # inputs = []
        # inputs_test = []
        # for i in range(batch_size):
        #     inputs_batch = []
        #     inputs_batch_test = []
        #     for j in range(n_input_steps):
        #         inputs_batch.append(np.zeros([int(n_steps / n_input_steps), self.n_inputs]) + random_steps[i, j])
        #         inputs_batch_test.append(np.zeros([int(n_steps / n_input_steps), self.n_inputs]) + random_steps_test[i, j])
        #     inputs_batch = np.concatenate(inputs_batch, axis=0)
        #     inputs_batch_test = np.concatenate(inputs_batch_test, axis=0)
        #     inputs.append(inputs_batch)
        #     inputs_test.append(inputs_batch_test)
        # input_train = np.array(inputs)
        # input_test = np.array(inputs_test)

        np.random.seed(0)
        input_train = np.random.randn(batch_size, n_steps, self.n_inputs)*0.4
        input_test = np.random.randn(batch_size, n_steps, self.n_inputs) * 0.4
        for i in range(n_steps-1):
            input_train[:,i+1] = 0.7 * input_train[:,i].copy() + 0.3 * input_train[:,i+1].copy()
            input_test[:, i + 1] = 0.7 * input_test[:, i].copy() + 0.3 * input_test[:, i + 1].copy()

        x_0 = mat([0., 0., 0.]).astype('float32')
        output_train = np.zeros([batch_size, n_steps, self.n_outputs])
        output_test = np.zeros([batch_size, n_steps, self.n_outputs])
        for i in range(batch_size):
            self.set_state(self.zero_state)
            for step in range(n_steps):
                output_train[i, step:step+1] = self.step(input_train[i, step:step+1])
            self.set_state(self.zero_state)
            for step in range(n_steps):
                output_test[i, step:step + 1] = self.step(input_test[i, step:step + 1])
        return input_train, input_test, output_train, output_test

    def generate_circle(self, batch_size, n_steps):
        inputs = np.ones([batch_size, n_steps, self.n_inputs])*0.2
        x_0 = x_0 = mat([0., 0., 0.]).astype('float32')
        output_list = []
        for i in range(batch_size):
            output_list.append(self.run(x_0, inputs[i], n_steps))
        return inputs, np.array(output_list)

    def generate_S(self, batch_size, n_steps):
        inputs = np.ones([batch_size, n_steps, self.n_inputs])*0.1
        inputs[:,:,-1] = np.stack([np.sin(np.arange(n_steps)*0.3) for i in range(batch_size)])
        x_0 = x_0 = mat([0., 0., 0.]).astype('float32')
        output_list = []
        for i in range(batch_size):
            output_list.append(self.run(x_0, inputs[i], n_steps))
        return inputs, np.array(output_list)

    def evaluate_performance(self, predictions, true_output):
        p_idx = []
        for i in range(self.n_outputs):
            p_idx.append(100. * (1. - np.sqrt(np.sum((np.square(predictions[:, :, i] - true_output[:, :, i])) / np.sum(
                np.square(np.mean(true_output[:, :, i]) - true_output[:, :, i]))))))
        return p_idx

    def visualize_results(self, predictions_train, output_train, predictions_test, output_test):
        time = np.arange(output_train.shape[1]) * self.Ts

        performance_idx = self.evaluate_performance(predictions_test, output_test)

        plt.figure(figsize=(15, 10))
        plt.title('Training')
        for i in range(output_train.shape[-1]):
            plt.subplot(output_train.shape[-1], 1, i+1)
            plt.plot(time, predictions_train[0,:,i], label='predicted')
            plt.plot(time, output_train[0,:,i], label='real')
            plt.legend()
            plt.ylabel('state {}'.format(i))
        plt.xlabel('Time [s]')
        plt.show()

        plt.figure(figsize=(8, 6)).suptitle('State predictions, ESN, fit: {:.2f}'.format(performance_idx))
        # plt.title('Test')
        for i in range(output_test.shape[-1]):
            plt.subplot(output_train.shape[-1], 1, i+1)
            plt.plot(time, predictions_test[0,:,i], label='predicted')
            plt.plot(time, output_test[0,:,i], label='real')
            plt.legend()
            if i == 0:
                plt.ylabel('Orizontal\nposition'.format(i))
            if i == 1:
                plt.ylabel('Vertical\nposition'.format(i))
            if i == 2:
                plt.ylabel('Angle\ncosine'.format(i))
            if i == 3:
                plt.ylabel('Angle\nsine'.format(i))
        plt.xlabel('Time [s]')
        plt.show()

# def AUV_model(initial_state, inputs, n_steps, Ts=0.5):
#
#     # Input shape [n_steps, 2]
#     # Initial state shape [1, 3]
#     x = initial_state.copy()
#     inputs = np.reshape(inputs, [inputs.shape[0], inputs.shape[1], 1])
#     state_list = []
#     for step in range(n_steps):
#         B_matrix = mat([[np.cos(x[0, 2]), 0.],
#                         [0., np.sin(x[0, 2])],
#                         [0.,        1.]]).reshape(3, 2)
#         d_state = (Ts * (B_matrix @ inputs[step])).reshape(x.shape)
#         x += d_state
#         state_list.append(x.copy())
#
#     return np.reshape(mat(state_list), [n_steps, x.shape[1]])


class Two_tank_model:
    def __init__(self, Ts=10):
        self.name = 'Two_tank_model'
        self.Ts = Ts
        self.n_inputs = 1
        self.n_outputs = 1
        self.zero_state = mat([-0.432, 0.528, 14., 7.])  # state is defined as initial states and initial output
        self.current_state = self.zero_state.copy()

    def set_state(self, state):
        self.current_state = state.copy()

    def run(self, initial_state, inputs, disturbance, n_steps):
        # Input shape [n_steps, n_inputs]
        # Initial state shape [1, 3]

        ''' Input is q3, state are [Wa4, Wb4, h] reaction invariants at out flow and tank level
            State measurement units: x1,x2: mL/s  x3: cm '''

        # equilibrium point
        x_0 = mat([-0.436, 0.5276, 13.746]).astype('float32')
        y_0 = mat([7.025])
        u_ = 15.6
        dist_ = 0.55

        # constants
        A1 = 207
        z = 11.5
        n = 0.607
        q4nominal = 32.8

        # nominal values at steady state
        pK1 = 6.35  # 1 and 2 dissociation constant of H2CO3
        pK2 = 10.25
        x1_ss = -0.436
        x2_ss = 0.5276
        x3_ss = 13.746
        y_ss = 7.025
        # Cv4 = q4nominal / ((x3_ss + z) ** n)
        Cv4 = 4.59
        q1 = 16.6
        Wa1 = 3.
        Wa2 = -30.
        Wa3 = -3.05
        Wb1 = 0.
        Wb2 = 30.
        Wb3 = 0.05

        x = initial_state[:, :3].copy()
        y = initial_state[:, 3:].copy()
        state_list = []
        output_list = []
        for step in range(n_steps):
            xd = x.copy()
            d = disturbance[step, :]
            ua = inputs[step, :]
            # evaluating state derivatives
            xd[:, 0] = (1 / (A1 * x[:, 2]) * (q1 * (Wa1 - x[:, 0]) + d * (Wa2 - x[:, 0]) + ua * (Wa3 - x[:, 0])))
            xd[:, 1] = (1 / (A1 * x[:, 2])) * (q1 * (Wb1 - x[:, 1]) + d * (Wb2 - x[:, 1]) + ua * (Wb3 - x[:, 1]))
            xd[:, 2] = (1 / A1) * (q1 + d + ua - Cv4 * (x[:, 2] + z) ** n)
            # evaluating output derivatives as constraint
            dcx2 = (1 + 2 * 10 ** (y - pK2)) / (1 + 10 ** (pK1 - y) + 10 ** (y - pK2))
            dcy = np.log(10) * (10 ** (y - 14) + 10 ** (-y) + x[:, 1] * (
                    10 ** (pK1 - y) + 10 ** (y - pK2) + 4 * 10 ** (pK1 - y) * 10 ** (y - pK2)) / (
                                        (1 + 10 ** (pK1 - y) + 10 ** (y - pK2)) ** 2))
            yd = -dcy ** -1 * (xd[:, 0] + dcx2 * xd[:, 1])
            # updating states and output with forward euler
            x += self.Ts * xd
            y += self.Ts * yd
            state_list.append(np.concatenate([x[0], y[0]]))
            output_list.append(y[0].copy())
            state_list = np.array(state_list)
            output_list = np.array(output_list)

        return state_list, output_list

    def run_2(self, t, initial_state, inputs, disturbance):

        # Input shape [n_steps, n_inputs]
        # Initial state shape [1, 3]

        ''' Input is q3, state are [Wa4, Wb4, h] reaction invariants at out flow and tank level
            State measurement units: x1,x2: mL/s  x3: cm '''

        # equilibrium point
        x_0 = mat([-0.436, 0.5276, 13.746]).astype('float32')
        y_0 = mat([7.025])
        u_ = 15.6
        dist_ = 0.55

        # constants
        A1 = 207
        z = 11.5
        n = 0.607
        q4nominal = 32.8

        # nominal values at steady state
        pK1 = 6.35  # 1 and 2 dissociation constant of H2CO3
        pK2 = 10.25
        x1_ss = -0.436
        x2_ss = 0.5276
        x3_ss = 13.746
        y_ss = 7.025
        # Cv4 = q4nominal / ((x3_ss + z) ** n)
        Cv4 = 4.59
        q1 = 16.6
        Wa1 = 3.
        Wa2 = -30.
        Wa3 = -3.05
        Wb1 = 0.
        Wb2 = 30.
        Wb3 = 0.05

        initial_state = np.reshape(initial_state, [1, 4])

        x = initial_state[:, :3].copy()
        y = initial_state[:, 3:].copy()
        state_list = []
        output_list = []
        xd = x.copy()
        d = disturbance[0, :]
        ua = inputs[0, :]
        # evaluating state derivatives
        xd[:, 0] = (1 / (A1 * x[:, 2]) * (q1 * (Wa1 - x[:, 0]) + d * (Wa2 - x[:, 0]) + ua * (Wa3 - x[:, 0])))
        xd[:, 1] = (1 / (A1 * x[:, 2])) * (q1 * (Wb1 - x[:, 1]) + d * (Wb2 - x[:, 1]) + ua * (Wb3 - x[:, 1]))
        xd[:, 2] = (1 / A1) * (q1 + d + ua - Cv4 * (x[:, 2] + z) ** n)
        # evaluating output derivatives as constraint
        dcx2 = (1 + 2 * 10 ** (y - pK2)) / (1 + 10 ** (pK1 - y) + 10 ** (y - pK2))
        dcy = np.log(10) * (10 ** (y - 14) + 10 ** (-y) + x[:, 1] * (
                10 ** (pK1 - y) + 10 ** (y - pK2) + 4 * 10 ** (pK1 - y) * 10 ** (y - pK2)) / (
                                        (1 + 10 ** (pK1 - y) + 10 ** (y - pK2)) ** 2))
        yd = -dcy ** -1 * (xd[:, 0] + dcx2 * xd[:, 1])

        state_der = np.concatenate([xd, yd], axis=-1)[0]

        return state_der

    def step(self, inputs, disturbance_value=0.55):
        state, output = self.run(self.current_state, inputs, disturbance=np.array([[disturbance_value]]), n_steps=1)
        system_output = self.current_state[:, 3:].copy()
        self.current_state = state.copy()
        return self.current_state.copy(), system_output

    def forward_(self, inputs, disturbance_value=0.55):
        state, output = self.run(self.current_state, inputs, disturbance=np.array([[disturbance_value]]), n_steps=1)
        self.current_state = state.copy()
        return self.current_state.copy(), output.copy()

    def forward(self, inputs, disturbance_value=0.55):
        run = partial(self.run_2, inputs=inputs, disturbance=np.array([[disturbance_value]]))

        sol = scipy.integrate.solve_ivp(run, [0, self.Ts], self.current_state[0])

        state = np.reshape(sol.y[:,-1], [1, 4])
        output = np.reshape(sol.y[-1:, -1], [1, self.n_outputs])
        self.current_state = state.copy()
        return self.current_state.copy(), output.copy()

    def generate_data(self, batch_size, n_steps):
        # n_input_steps = int(n_steps/(1000/self.Ts))  # one step every 1000 s
        # n_dist_steps = int(n_steps/(50/self.Ts))  # one step every 5000 s
        # init_states_train = []
        # init_states_test = []
        # np.random.seed(777)
        # random_steps = (np.random.rand(batch_size, n_input_steps) - 0.5) * 4
        # random_steps_test = (np.random.rand(batch_size, n_input_steps) - 0.5) * 4
        # random_steps_dist_train = (np.random.rand(batch_size, n_dist_steps) - 0.5) * 0.
        # random_steps_dist_test = (np.random.rand(batch_size, n_dist_steps) - 0.5) * 0.
        # inputs = []
        # inputs_test = []
        # dist_train = []
        # dist_test = []
        # for i in range(batch_size):
        #     inputs_batch = []
        #     inputs_batch_test = []
        #     dist_batch_train = []
        #     dist_batch_test = []
        #     for j in range(n_input_steps):
        #         inputs_batch.append(np.ones([int(n_steps / n_input_steps), self.n_inputs]) * 14.7 + random_steps[i, j])
        #         inputs_batch_test.append(np.ones([int(n_steps / n_input_steps), self.n_inputs]) * 14.7 + random_steps_test[i, j])
        #     for k in range(n_dist_steps):
        #         dist_batch_train.append(
        #             np.ones([int(n_steps / n_dist_steps), 1]) * 0.55 + random_steps_dist_train[i, k])
        #         dist_batch_test.append(
        #             np.ones([int(n_steps / n_dist_steps), 1]) * 0.55 + random_steps_dist_test[i, k])
        #     inputs_batch = np.concatenate(inputs_batch, axis=0)
        #     inputs_batch_test = np.concatenate(inputs_batch_test, axis=0)
        #     dist_batch_train = np.concatenate(dist_batch_train, axis=0)
        #     dist_batch_test = np.concatenate(dist_batch_test, axis=0)
        #     inputs.append(inputs_batch)
        #     inputs_test.append(inputs_batch_test)
        #     dist_train.append(dist_batch_train)
        #     dist_test.append(dist_batch_test)
        # input_train = np.array(inputs) + np.random.randn(batch_size, n_steps, self.n_inputs)*0.1
        # input_test = np.array(inputs_test)
        # dist_train = np.array(dist_train)
        # dist_test = np.array(dist_test)
        #
        # output_train = np.zeros([batch_size, n_steps, 1])
        # output_test = np.zeros([batch_size, n_steps, 1])
        # for i in range(batch_size):
        #     state = self.zero_state # + np.concatenate([np.random.randn(3)*0.02, np.random.randn(1)*0.1])
        #     init_states_train.append(state[0, 3:].copy())
        #     self.set_state(state)
        #     for step in range(n_steps):
        #         state, output_train[i, step:step+1] = self.forward(input_train[i, step:step+1])
        #     state = self.zero_state # + np.concatenate([np.random.randn(3)*0.02, np.random.randn(1)*0.1])
        #     init_states_test.append(state[0, 3:].copy())
        #     self.set_state(state)
        #     for step in range(n_steps):
        #         state, output_test[i, step:step+1] = self.forward(input_test[i, step:step+1])
        # init_states_train = np.array(init_states_train)
        # init_states_test = np.array(init_states_test)
        np.random.seed(777)

        step_length = 100

        input_train = []
        output_train = []
        input_test = []
        output_test = []

        for batch in range(batch_size):
            input_series_train = np.concatenate([np.ones([step_length, 1])*14.7 + np.random.rand()*4. - 2. for _ in range(int(n_steps/step_length) + 1)])
            input_series_test = np.concatenate([np.ones([step_length, 1])*14.7 + np.random.rand()*4. - 2. for _ in range(int(n_steps/step_length) + 1)])

            input_series_train, input_series_test = input_series_train[:n_steps], input_series_test[:n_steps]

            input_series_train += np.random.randn(n_steps, 1) * 0.1
            input_series_train = np.clip(input_series_train, 12.7, 16.7)

            input_series_test += np.random.randn(n_steps, 1) * 0.1
            input_series_test = np.clip(input_series_test, 12.7, 16.7)

            self.set_state(self.zero_state)
            output_series_train = []
            for step in range(n_steps):
                _, out = self.forward(input_series_train[step:step+1])
                output_series_train.append(out[0])
            output_series_train = np.array(output_series_train)

            self.set_state(self.zero_state)
            output_series_test = []
            for step in range(n_steps):
                _, out = self.forward(input_series_test[step:step + 1])
                output_series_test.append(out[0])
            output_series_test = np.array(output_series_test)

            input_train.append(input_series_train)
            input_test.append(input_series_test)

            output_train.append(output_series_train)
            output_test.append(output_series_test)

        input_train = np.array(input_train)
        input_test = np.array(input_test)

        output_train = np.array(output_train)
        output_test = np.array(output_test)

        init_states_train = np.ones([batch_size, self.n_outputs]) * 7.025
        init_states_test = np.ones([batch_size, self.n_outputs]) * 7.025

        return input_train, input_test, output_train, output_test, init_states_train, init_states_test

    def evaluate_performance(self, predictions, true_output):
        p_idx = []
        for i in range(self.n_outputs):
            p_idx.append(100. * (1. - np.sqrt(np.sum((np.square(predictions[:,:,i] - true_output[:,:,i])) / np.sum(np.square(np.mean(true_output[:,:,i]) - true_output[:,:,i]))))))
        return p_idx

    def visualize_results(self, predictions_train, output_train, predictions_test, output_test, network):
        time = np.arange(output_train.shape[1]) * self.Ts

        performance_idx = self.evaluate_performance(predictions_test, output_test)
        # washout = network.sess.run(network.washout)*self.Ts

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_title('pH predictions on training set, {} network'.format(network.name))
        ax.plot(time, predictions_train[0], label='predicted')
        ax.plot(time, output_train[0], label='real')
        # ax.plot(np.ones(100)*washout, np.linspace(0, 10, num=100), '--', label='washout limit')
        ax.legend()
        ax.axis([0, time[-1], 5.5, 8.])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('pH')
        ax.grid()

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if network.observer_name is not None:
            ax.set_title('pH predictions on test set, {} network + {} filter, fit: {:.2f} %'.format(network.name,
                                                                                                 network.observer_name,
                                                                                                 performance_idx[0]))
        else:
            ax.set_title('pH predictions on test set, {} network, fit: {:.2f} %'.format(network.name, performance_idx[0]))
        ax.plot(time, predictions_test[0], label='predicted')
        ax.plot(time, output_test[0], label='real')
        ax.legend()
        ax.axis([0, time[-1], 5.5, 8.])
        ax.set_xlabel('Time')
        ax.set_ylabel('pH')
        ax.grid()


class CSTR:
    def __init__(self, Ts=0.1):
        self.name = 'CSTR'
        self.n_inputs = 2
        self.n_outputs = 2
        self.zero_state = np.array([0.692, 0.287])
        self.current_state = self.zero_state.copy()
        self.current_inputs = np.array([0.8, 0.8])
        self.Ts = Ts

    def set_state(self, state):
        self.current_state = state.copy()

    def set_inputs(self, inputs):
        self.current_inputs = inputs.copy()

    def run(self, t, state):

        # returns dx/dt = run(x, u)

        q = self.current_inputs[0]   # Input flow  [m^3/s]
        T = self.current_inputs[1]   # Tank temperature  [K]
        C_A0 = 0.8     # Concentration of A in input feed

        C_A = state[0]
        C_R = state[1]

        k0_list = [1.0, 0.7, 0.1, 0.006]   # Arrhenius pre-exponentials constants
        E = [8.33, 10.0, 50.0, 83.3]       # Normalized activation energies

        k = []
        for idx, k0 in enumerate(k0_list):
            k.append(k0 * np.exp(-E[idx] * (1/T - 1)))
        dC_A = q * (C_A0 - C_A) - k[0] * C_A + k[3] * C_R

        dC_R = q * (1 - C_A0 - C_R) + k[0] * C_A + k[2] * (1 - C_A - C_R) - (k[1] + k[3]) * C_R

        return dC_A, dC_R

    # def step(self, input_series):
    #
    #     outputs = []
    #
    #     self.set_inputs(input_series[0])
    #     sim_output = spi.solve_ivp(self.run, t_span=(0., self.Ts), y0=self.current_state, method='RK23', min_step=self.Ts/100)
    #     self.current_state = sim_output.y[:,-1].copy()
    #     outputs.append(sim_output.y[:,0].copy())
    #
    #     return self.current_state.copy(), np.array(outputs)
    #
    # def forward(self, input_series):
    #
    #     outputs = []
    #
    #     self.set_inputs(input_series[0])
    #     sim_output = spi.solve_ivp(self.run, t_span=(0., self.Ts), y0=self.current_state, method='RK23', min_step=self.Ts/100)
    #     self.current_state = sim_output.y[:,-1].copy()
    #     outputs.append(sim_output.y[:,-1].copy())
    #
    #     return self.current_state.copy(), np.array(outputs)

    def step(self, input_series, time_divisions=100):

        self.set_inputs(input_series[0])
        for step in range(time_divisions):
            output = self.current_state.copy()
            self.current_state = self.current_state.copy() + (self.Ts/time_divisions)*np.array(self.run(0., self.current_state)).copy()
        return self.current_state.copy(), np.reshape(output, [1, self.n_outputs])

    def forward(self, input_series, time_divisions=100):

        self.set_inputs(input_series[0])
        for step in range(time_divisions):
            output = self.current_state.copy()
            self.current_state = self.current_state.copy() + (self.Ts/time_divisions)*np.array(self.run(0., self.current_state)).copy()
        return self.current_state.copy(), np.reshape(self.current_state, [1, self.n_outputs])

    def generate_data(self, batch_size, n_steps, flow_period=50):

        np.random.seed(150)

        triangular_window = (signal.bartlett(flow_period+1, sym=True)*0.35 + 0.7)[:-1]

        inputs_train_list = []
        outputs_train_list = []
        inputs_test_list = []
        outputs_test_list = []

        for batch in range(batch_size):

            output_train_list_batch = []
            output_test_list_batch = []

            inputs_train = np.zeros([n_steps, self.n_inputs])
            temperature_steps_value_train = [np.random.rand()*0.35 + 0.7 for i in range(int(n_steps/flow_period) + 1)]
            inputs_test = np.zeros([n_steps, self.n_inputs])
            temperature_steps_value_test = [np.random.rand()*0.35 + 0.7 for i in range(int(n_steps / flow_period) + 1)]

            # Generating triangular wave for flow input and steps for temperature input

            for i in range(int(n_steps/flow_period)):
                inputs_train[i*flow_period:(i+1)*flow_period, 0] = triangular_window.copy()
                inputs_train[i*flow_period:(i+1)*flow_period, 1] = np.ones([flow_period]) * temperature_steps_value_train[i]
                inputs_test[i * flow_period:(i + 1) * flow_period, 0] = triangular_window.copy()
                inputs_test[i * flow_period:(i + 1) * flow_period, 1] = np.ones([flow_period]) * temperature_steps_value_test[i]
            if n_steps < flow_period:
                i = -1
            if (i+1)*flow_period < n_steps:
                inputs_train[(i+1)*flow_period: n_steps, 0] = triangular_window[:n_steps-(i+1)*flow_period].copy()
                inputs_train[(i+1)*flow_period: n_steps, 1] = np.ones([n_steps-(i+1)*flow_period]) * temperature_steps_value_train[i+1]
                inputs_test[(i + 1) * flow_period: n_steps, 0] = triangular_window[:n_steps-(i+1)*flow_period].copy()
                inputs_test[(i + 1) * flow_period: n_steps, 1] = np.ones([n_steps-(i+1)*flow_period])*temperature_steps_value_test[i+1]

            inputs_train_list.append(inputs_train)
            inputs_test_list.append(inputs_test)

            self.set_state(np.random.rand(2))

            for step in range(n_steps):
                state, output = self.step(inputs_train[step:step+1])
                output_train_list_batch.append(output[0].copy())

            self.set_state(np.random.rand(2))

            for step in range(n_steps):
                state, output = self.step(inputs_test[step:step+1])
                output_test_list_batch.append(output[0].copy())

            outputs_train_list.append(np.array(output_train_list_batch))
            outputs_test_list.append(np.array(output_test_list_batch))

        return np.array(inputs_train_list).copy(), np.array(inputs_test_list).copy(), \
               np.array(outputs_train_list).copy(), np.array(outputs_test_list).copy()

    def evaluate_performance(self, predictions, true_output):
        p_idx = []
        for i in range(self.n_outputs):
            p_idx.append(100. * (1. - np.sqrt(np.sum((np.square(predictions[:,:,i] - true_output[:,:,i])) / np.sum(np.square(np.mean(true_output[:,:,i]) - true_output[:,:,i]))))))
        return p_idx

    def visualize_results(self, predictions_train, output_train, predictions_test, output_test, network):
        time = np.arange(output_train.shape[1]) * self.Ts

        performance_idx = self.evaluate_performance(predictions_test, output_test)

        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        ax[0].set_title('C_a predictions on training set, {} network'.format(network.name))
        ax[0].plot(time, predictions_train[0,:,0], label='predicted C_a')
        ax[0].plot(time, output_train[0,:,0], label='real C_a')
        ax[0].legend()
        ax[0].set_ylabel('Concentration')
        ax[1].set_title('C_r predictions on training set')
        ax[1].plot(time, predictions_train[0, :, 1], label='predicted C_r')
        ax[1].plot(time, output_train[0, :, 1], label='real C_r')
        ax[1].legend()
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Concentration')

        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        if network.observer_name is not None:
            ax[0].set_title('C_a predictions on test set, {} network + {} filter, fit: {:.2f} %'.format(network.name,
                                                                                                 network.observer_name,
                                                                                                 performance_idx[0]))
            ax[1].set_title('C_r predictions on test set, {} network + {} filter, fit: {:.2f} %'.format(network.name,
                                                                                                 network.observer_name,
                                                                                                 performance_idx[1]))
        else:
            ax[0].set_title('C_a predictions on test set, {} network, fit: {:.2f} %'.format(network.name, performance_idx[0]))
            ax[1].set_title('C_r predictions on test set, {} network, fit: {:.2f} %'.format(network.name, performance_idx[1]))

        ax[0].plot(time, predictions_test[0,:,0], label='predicted C_a')
        ax[0].plot(time, output_test[0,:,0], label='real C_a')
        ax[0].legend()
        ax[0].set_ylabel('Concentration')
        ax[1].plot(time, predictions_test[0, :, 1], label='predicted C_r')
        ax[1].plot(time, output_test[0, :, 1], label='real C_r')
        ax[1].legend()
        ax[1].set_ylabel('Concentration')
        ax[1].set_xlabel('Time [s]')


class InvertedPendulum:
    """Inverted Pendulum.

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

    def __init__(self, mass=0.15, length=0.5, friction=0.1, dt=0.01,
                 normalization=[np.array([1., 1.]), np.array([1.])]):
        """Initialization; see `InvertedPendulum`."""
        super(InvertedPendulum, self).__init__()
        self.n_inputs = 1
        self.n_outputs = 1
        self.name = 'pendulum'
        self.mass = mass
        self.length = length
        self.gravity = 9.81
        self.friction = friction
        self.dt = dt
        self.Ts = dt
        self.zero_state = np.zeros([1,2])
        self.current_state = self.zero_state.copy()

        self.normalization = normalization
        if normalization is not None:
            self.normalization = [norm
                                  for norm in normalization]
            self.inv_norm = [norm ** -1 for norm in self.normalization]

    @property
    def inertia(self):
        """Return inertia of the pendulum."""
        return self.mass * self.length ** 2

    def normalize(self, state, action):
        """Normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
        state = np.matmul(state, Tx_inv)

        if action is not None:
            action = np.matmul(action, Tu_inv)

        return state, action

    def denormalize(self, state, action):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx, Tu = map(np.diag, self.normalization)

        state = np.matmul(state, Tx)
        if action is not None:
            action = np.matmul(action, Tu)

        return state, action

    def linearize(self):
        """Return the linearized system.

        Returns
        -------
        a : ndarray
            The state matrix.
        b : ndarray
            The action matrix.

        """
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        A = np.array([[0, 1],
                      [gravity / length, -friction / inertia]],
                     dtype=np.float32)

        B = np.array([[0],
                      [1 / inertia]],
                     dtype=np.float32)

        if self.normalization is not None:
            Tx, Tu = map(np.diag, self.normalization)
            Tx_inv, Tu_inv = map(np.diag, self.inv_norm)

            A = np.linalg.multi_dot((Tx_inv, A, Tx))
            B = np.linalg.multi_dot((Tx_inv, B, Tu))

        sys = signal.StateSpace(A, B, np.eye(2), np.zeros((2, 1)))
        sysd = sys.to_discrete(self.dt)
        return sysd.A, sysd.B

    def set_state(self, state):
        self.current_state = state.copy()

    def forward(self, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        states: ndarray or Tensor
            Unnormalized states.
        actions: ndarray or Tensor
            Unnormalized actions.

        Returns
        -------
        x_dot: Tensor
            The normalized derivative of the dynamics

        """
        # Physical dynamics
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        state = self.current_state.copy()

        state, action = self.denormalize(state, action)
        angle = state[:,0]
        angular_velocity = state[:,1]

        x_ddot = gravity / length * np.sin(angle) + action[0] / inertia

        if friction > 0:
            x_ddot -= friction / inertia * angular_velocity

        state_derivative = np.array([angular_velocity, x_ddot])
        # Normalize
        s = state + self.dt * state_derivative.reshape([1, 2])
        s = self.normalize(s, action)[0]
        angle_ = s[:,0]
        angular_velocity_ = s[:,1]

        state_next = np.array([angle_, angular_velocity_])
        state_next = state_next.reshape([1, 2])
        self.current_state = state_next.copy()

        output = state_next[:, 0:1].copy()
        return state_next.copy(), output

    def step(self, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        states: ndarray or Tensor
            Unnormalized states.
        actions: ndarray or Tensor
            Unnormalized actions.

        Returns
        -------
        x_dot: Tensor
            The normalized derivative of the dynamics

        """
        # Physical dynamics
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        state = self.current_state.copy()
        output = state.copy()

        state, action = self.denormalize(state, action)
        angle = state[:,0]
        angular_velocity = state[:,1]

        x_ddot = gravity / length * np.sin(angle) + action[0] / inertia

        if friction > 0:
            x_ddot -= friction / inertia * angular_velocity

        state_derivative = np.array([angular_velocity, x_ddot])
        # Normalize
        s = state + self.dt * state_derivative.reshape([1, 2])
        s = self.normalize(s, action)[0]
        angle_ = s[:,0]
        angular_velocity_ = s[:,1]

        state_next = np.array([angle_, angular_velocity_])
        state_next = state_next.reshape([1, 2])
        self.current_state = state_next.copy()
        return state_next, output

    def closed_loop_sim(self, x_0, K_, n_steps, noise_std=0.):
            x_ = x_0
            self.set_state(x_0)
            x_list = []
            y_list = []
            u_list = []
            for s in range(n_steps):

                if isinstance(K_ ,np.ndarray):
                    u =  np.clip(x_ @ (-K_) + noise_std * np.random.randn(), -1., 1.)
                elif isinstance(K_ ,nn.Module):
                    u = K_(x_)
                else:
                    print('error')
                    return
                u_list.append(u[0])
                x_, out = self.forward(u)
                x_list.append(x_[0])
                y_list.append(out[0])
            x_list = np.array(x_list)
            y_list = np.array(y_list)
            u_list = np.array(u_list)
            return u_list, x_list, y_list

    def generate_data(self, batch_size, n_steps, noise_std=0.2):
        np.random.seed(18)
        K = np.diag(self.normalization[0]) @ np.array([[10./(2*np.pi)], [0.]])

        input_train, input_test, output_train, output_test = [], [], [], []
        init_states_train = []
        init_states_test = []

        for batch in range(batch_size):

            if batch == int(batch_size/2):
                K = np.zeros([2, 1])

            if batch < int(batch_size/2):
                state = np.random.randn(1,2) * 0.2
            else:
                state = np.random.randn(1,2) * 0.2 + np.array([[np.random.choice([-0.5, 0.5]), 0.]])
                if batch > int(batch_size/1.1):
                    state = np.random.randn(1,2) * 0.05 + np.array([[np.random.choice([-0.5, 0.5]), 0.]])
                    noise_std = 0.05
            init_states_train.append(state[0].copy())
            inputs_batch, state_batch, outputs_batch = self.closed_loop_sim(state, K, n_steps, noise_std)
            input_train.append(inputs_batch)
            output_train.append(state_batch)

            if batch < int(batch_size / 2):
                state = np.random.randn(1, 2) * 0.2
            else:
                state = np.random.randn(1, 2) * 0.2 + np.array([[randint(0, 1) - 0.5, 0.]])
                if batch > int(batch_size/1.1):
                    state = np.random.randn(1,2) * 0.05 + np.array([[randint(0, 1)-0.5, 0.]])
                    noise_std = 0.05
            init_states_test.append(state[0].copy())
            inputs_batch, state_batch, outputs_batch = self.closed_loop_sim(state, K, n_steps, noise_std)
            input_test.append(inputs_batch)
            output_test.append(state_batch)

        input_train = np.array(input_train)
        input_test = np.array(input_test)
        states_train = np.array(output_train)
        states_test = np.array(output_test)
        output_train = states_train[:, :, 0:1].copy()
        output_test = states_test[:, :, 0:1].copy()
        init_states_train = np.array(init_states_train)
        init_states_test = np.array(init_states_test)

        return input_train, states_train, output_train, input_test, states_test, output_test, init_states_train, init_states_test

    def evaluate_performance(self, predictions, true_output):
        p_idx = []
        for i in range(self.n_outputs):
            p_idx.append(100. * (1. - np.sqrt(np.sum((np.square(predictions[:,:,i] - true_output[:,:,i])) / np.sum(np.square(np.mean(true_output[:,:,i]) - true_output[:,:,i]))))))
        return p_idx

    def visualize_results(self, predictions_train, output_train, predictions_test, output_test, network):
        time = np.arange(output_train.shape[1]) * self.Ts

        performance_idx = self.evaluate_performance(predictions_test, output_test)

        # fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        # ax[0].set_title('Angle predictions on training set')
        # ax[0].plot(time, predictions_train[0,:,0], label='predicted')
        # ax[0].plot(time, output_train[0,:,0], label='real')
        # ax[0].legend()
        # ax[0].set_ylabel('Angle')
        # ax[1].set_title('Angular velocity predictions on training set')
        # ax[1].plot(time, predictions_train[0, :, 1], label='predicted')
        # ax[1].plot(time, output_train[0, :, 1], label='real')
        # ax[1].legend()
        # ax[1].set_xlabel('Time [s]')
        # ax[1].set_ylabel('Angular velocity')
        #
        # fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        # if network.observer_name is not None:
        #     ax[0].set_title('Angle predictions on test set, {} network + {} filter, fit: {:.2f} %'.format(network.name,
        #                                                                                          network.observer_name,
        #                                                                                          performance_idx[0]))
        #     ax[1].set_title('Angular velocity predictions on test set, {} network + {} filter, fit: {:.2f} %'.format(network.name,
        #                                                                                          network.observer_name,
        #                                                                                          performance_idx[1]))
        # else:
        #     ax[0].set_title('Angle predictions on test set, {} network, fit: {:.2f} %'.format(network.name, performance_idx[0]))
        #     ax[1].set_title('Angular velocity predictions on test set, {} network, fit: {:.2f} %'.format(network.name, performance_idx[1]))
        #
        # ax[0].plot(time, predictions_test[0,:,0], label='predicted')
        # ax[0].plot(time, output_test[0,:,0], label='real')
        # ax[0].legend()
        # ax[0].set_ylabel('Angle')
        # ax[1].plot(time, predictions_test[0, :, 1], label='predicted')
        # ax[1].plot(time, output_test[0, :, 1], label='real')
        # ax[1].legend()
        # ax[1].set_ylabel('Angle')
        # ax[1].set_xlabel('Time [s]')

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_title('Angle predictions on training set')
        ax.plot(time, predictions_train[0, :, 0], label='predicted')
        ax.plot(time, output_train[0, :, 0], label='real')
        ax.legend()
        ax.set_ylabel('Angle')

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        if network.observer_name is not None:
            ax.set_title('Angle predictions on test set, {} network + {} filter, fit: {:.2f} %'.format(network.name,
                                                                                                       network.observer_name,
                                                                                                       performance_idx[
                                                                                                           0]))
        else:
            ax.set_title(
                'Angle predictions on test set, {} network, fit: {:.2f} %'.format(network.name, performance_idx[0]))

        ax.plot(time, predictions_test[0, :, 0], label='predicted')
        ax.plot(time, output_test[0, :, 0], label='real')
        ax.legend()
        ax.set_ylabel('Angle')


class DoublePendulum:
    def __init__(self, m1=0.2, m2=0.2, L1=0.5, L2=0.5, d1=0.25, d2=0.25, c1=0.1, c2=0.1, Ts=0.01, saturation=[np.array([-1, 1]) for _ in range(2)]):
        """Double pendulum initialization"""
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.d1 = d1
        self.d2 = d2
        self.c1 = c1
        self.c2 = c2
        self.saturation = saturation
        self.gravity = 9.81
        self.zero_state = np.zeros([1, 4])
        self.current_state = self.zero_state.copy()
        self.n_inputs = 2
        self.n_outputs = 2
        self.Ts = Ts

    def saturate(self, input_value):
        inputs_list = np.split(input_value, indices_or_sections=self.n_inputs, axis=-1)
        for i, input in enumerate(inputs_list):
            inputs_list[i] = np.clip(input, self.saturation[i][0], self.saturation[i][1])
        input_norm = np.concatenate(inputs_list, axis=-1)
        return input_norm

    @property
    def I1(self):
        return 1/12 * self.m1 * self.L1 ** 2

    @property
    def I2(self):
        return 1/12 * self.m2 * self.L2 ** 2

    def set_state(self, state):
        """Setting the current state of the pendulum.

        The state is a numpy array of dimension (1, 4)
        defined as [theta_1, theta_1d, theta_2, theta_2d]"""

        self.current_state = state.copy()

    def direct_kinematics(self, angles):
        x = -self.L1 * np.sin(angles[0]) - self.L2 * np.sin(angles[0] + angles[1])
        y = self.L1 * np.cos(angles[0]) + self.L2 * np.cos(angles[0] + angles[1])
        return np.array([x, y])

    def inverse_kinematics(self, position_xy):
        costheta_2 = (position_xy[0] ** 2 + position_xy[1] ** 2 - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
        sintheta_2 = np.sqrt(1. - np.square(costheta_2))
        theta_2 = np.arctan2(sintheta_2, costheta_2)
        theta_1 = np.arctan2(position_xy[1], position_xy[0]) - np.arctan2(self.L2 * sintheta_2, self.L1 + self.L2*costheta_2) - np.pi/2
        return np.array([[theta_1, theta_2]])

    def forward(self, inputs, dt=None):
        """Computing the next state and measured output given an input.

        inputs: numpy array (1, 2)"""

        inputs = self.saturate(inputs)

        if dt is None:
            dt = self.Ts

        current_state = self.current_state.copy()

        # Solves the ode
        new_state = solve(self.get_state_derivatives, current_state, inputs, dt)

        self.current_state = new_state.copy()

        theta_1 = new_state[:, 0]
        theta_2 = new_state[:, 2]

        output = np.transpose(np.array([theta_1, theta_2]))

        return new_state, output

    def get_state_derivatives(self, t, initial_state, input_value):
        """Returns the derivative of the state used by the ode solver"""

        m1 = self.m1
        m2 = self.m2
        L1 = self.L1
        L2 = self.L2
        d1 = self.d1
        d2 = self.d2
        c1 = self.c1
        c2 = self.c2
        g = self.gravity
        I1 = self.I1
        I2 = self.I2

        theta_1 = initial_state[0]
        theta_1d = initial_state[1]
        theta_2 = initial_state[2]
        theta_2d = initial_state[3]

        tau_1 = input_value[0, 0]
        tau_2 = input_value[0, 1]

        A = np.zeros([2, 2])

        A[0, 0] = I1 + m1*d1**2 + I2 + m2*L1**2 + m2*d2**2 + 2*m2*L1*d2*np.cos(theta_2)
        A[0, 1] = I2 + m2*d2**2 + m2*L1*d2*np.cos(theta_2)
        A[1, 0] = I2 + m2*d2**2 + m2*L1*d2*np.cos(theta_2)
        A[1, 1] = I2 + m2*d2**2

        B = np.zeros([2, 1])

        B[0, 0] = tau_1 - c1*theta_1d + m2*g*d2*np.sin(theta_1 + theta_2) + m2*g*L1*np.sin(theta_1) + \
                  m1*g*d1*np.sin(theta_1) + m2*L1*d2*theta_2d**2*np.sin(theta_2) + \
                  2*m2*L1*d2*theta_1d*theta_2d*np.sin(theta_2)
        B[1, 0] = tau_2 - c2*theta_2d + m2*g*d2*np.sin(theta_1 + theta_2) - m2*L1*d2*theta_1d**2*np.sin(theta_2)

        theta_1dd = (np.linalg.inv(A) @ B)[0, 0]
        theta_2dd = (np.linalg.inv(A) @ B)[1, 0]

        state_derivative = np.array([theta_1d, theta_1dd, theta_2d, theta_2dd])

        return state_derivative

    def simulate_free(self, initial_state, steps, dt=None, noise_std=0.):
        """Run a simulation of the free falling pendulum.

        initial_state can be a list of states, in which case a simulation
        will be run for each initial state

        Returns a numpy array of the state time-series of shape (n_simulations, steps, 4)"""

        if dt is None:
            dt = self.Ts

        # if not isinstance(initial_state, list):
        #     initial_state = [initial_state]

        # t1 = []
        # t1d = []
        # t2 = []
        # t2d = []
        #
        # plt.figure()
        # plt.plot([0.], [0.], '*', label='hinge')

        # for state in initial_state:

            # x1_list = []
            # y1_list = []
            # x2_list = []
            # y2_list = []
            #
            # theta_1_list = []
            # theta_1d_list = []
            # theta_2_list = []
            # theta_2d_list = []

        inputs = []
        states = []
        outputs = []

        self.set_state(initial_state)

        for step in range(steps):

            input_value = np.random.randn(1, 2)*noise_std
            input_value = self.saturate(input_value)

            state, output = self.forward(input_value, dt=dt)

            inputs.append(input_value[0])
            states.append(state[0])
            outputs.append(output[0])

            #     x1 = - self.L1 * np.sin(output[:, 0])
            #     y1 = self.L1 * np.cos(output[:, 0])
            #     x2 = - self.L1 * np.sin(output[:, 0]) - self.L2 * np.sin(output[:, 0] + output[:, 1])
            #     y2 = self.L1 * np.cos(output[:, 0]) + self.L2 * np.cos(output[:, 0] + output[:, 1])
            #     pos = [x1, y1, x2, y2]
            #
            #     x1_list.append(pos[0])
            #     y1_list.append(pos[1])
            #     x2_list.append(pos[2])
            #     y2_list.append(pos[3])
            #
            #     theta_1_list.append(state[:, 0])
            #     theta_1d_list.append(state[:, 1])
            #     theta_2_list.append(state[:, 2])
            #     theta_2d_list.append(state[:, 3])
            #
            # plot_limits = (self.L1 + self.L2) * 1.1
            # plt.plot(x1_list, y1_list, label='first joint position')
            # plt.plot(x2_list, y2_list, label='second joint position')
            # plt.grid()
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.axis([-plot_limits, plot_limits, -plot_limits, plot_limits])
            # # plt.legend()
            #
            # theta_1 = np.array(theta_1_list)
            # theta_1d = np.array(theta_1d_list)
            # theta_2 = np.array(theta_2_list)
            # theta_2d = np.array(theta_2d_list)
            #
            # t1.append(theta_1)
            # t2.append(theta_2)
            # t1d.append(theta_1d)
            # t2d.append(theta_2d)

        # plt.show()
        inputs = np.array(inputs)
        states = np.array(states)
        outputs = np.array(outputs)

        return states, inputs, outputs

    def linearize(self, x_0, input_value, eps=1e-6):
        """Retrieves the linearized model matrices used to tune the LQR"""

        current_state = x_0.copy()

        A_columns = []

        for i in range(4):
            delta_state = np.zeros([1, 4])
            delta_state[:, i] = -eps/2
            self.set_state(current_state + delta_state)
            new_state_neg, output = self.forward(input_value)
            delta_state = np.zeros([1, 4])
            delta_state[:, i] = eps/2
            self.set_state(current_state + delta_state)
            new_state_pos, output = self.forward(input_value)
            state_der = (new_state_pos - new_state_neg) / eps
            A_columns.append(np.transpose(state_der))

        A = np.concatenate(A_columns, axis=-1)

        B_columns = []

        for i in range(self.n_inputs):
            self.set_state(current_state)
            delta_input = np.zeros([1, 2])
            delta_input[:, i] = -eps/2
            new_state_neg, output = self.forward(input_value + delta_input)
            self.set_state(current_state)
            delta_input = np.zeros([1, 2])
            delta_input[:, i] = eps/2
            new_state_pos, output = self.forward(input_value + delta_input)
            state_der = (new_state_pos - new_state_neg) / eps
            B_columns.append(np.transpose(state_der))

        self.set_state(current_state)

        B = np.concatenate(B_columns, axis=-1)

        C = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.]])

        return A, B, C

    def closed_loop_sim(self, x_0, steps, K=None, noise_std=0., target=np.array([[0., 0., 0., 0.]])):
        """Perform a closed loop simulation using LQR.

        Noise can be added to the control input using noise_std"""

        if K is None:
            A, B, C = self.linearize(np.zeros([1, 4]), np.zeros([1, 2]))

            Q = np.diag([10., 0.1, 10., 0.1])
            R = np.diag([2., 2.])

            K, P, eig = dlqr(A, B, Q, R)

        output_list = []
        input_list = []
        state_list = []

        self.set_state(x_0.copy())
        for step in range(steps):
            input_value = -(self.current_state - target) @ np.transpose(K) + np.random.randn(1, 2)*noise_std
            input_value = self.saturate(input_value)
            state, output = self.forward(input_value)

            input_list.append(input_value[0].copy())
            output_list.append(output[0].copy())
            state_list.append(state[0].copy())

        input_list = np.array(input_list)
        output_list = np.array(output_list)
        state_list = np.array(state_list)

        return state_list, input_list, output_list

    def generate_data(self, batch_size, n_steps, noise_std=0.5):

        np.random.seed(11111)

        input_list_train = []
        output_list_train = []
        state_list_train = []
        input_list_test = []
        output_list_test = []
        state_list_test = []

        targets = np.meshgrid([0, -np.pi, np.pi], [0.], [0., -np.pi, np.pi], [0])
        targets = np.reshape(np.array(targets).T, [9, 1, 4])

        init_states_train = [np.pi*(np.random.rand(1, 4)*2 - 1) for _ in range(batch_size)]

        i = 0
        for batch in range(batch_size):
            target = targets[i]
            i = (i+1)%targets.shape[0]

            if batch < batch_size/2:
                states, inputs, outputs = self.closed_loop_sim(init_states_train[batch], steps=n_steps,
                                                               target=target, noise_std=noise_std)
                input_list_train.append(inputs)
                output_list_train.append(outputs)
                state_list_train.append(states)

            else:
                states, inputs, outputs = self.simulate_free(init_states_train[batch], steps=n_steps, noise_std=noise_std)

                input_list_train.append(inputs)
                output_list_train.append(outputs)
                state_list_train.append(states)

        init_states_test = [np.pi * (np.random.rand(1, 4) * 2 - 1) for _ in range(batch_size)]

        i = 0
        for batch in range(batch_size):
            target = targets[i]
            i = (i + 1) % targets.shape[0]

            if batch < batch_size / 2:
                states, inputs, outputs = self.closed_loop_sim(init_states_test[batch], steps=n_steps,
                                                               target=target, noise_std=noise_std)
                input_list_test.append(inputs)
                output_list_test.append(outputs)
                state_list_test.append(states)

            else:
                states, inputs, outputs = self.simulate_free(init_states_test[batch], steps=n_steps,
                                                             noise_std=noise_std)

                input_list_test.append(inputs)
                output_list_test.append(outputs)
                state_list_test.append(states)

        for i, state in enumerate(init_states_train):
            init_states_train[i] = state[0]

        for i, state in enumerate(init_states_test):
            init_states_test[i] = state[0]

        init_states_train = np.array(init_states_train)
        init_states_test = np.array(init_states_test)
        input_list_train = np.array(input_list_train)
        state_list_train = np.array(state_list_train)
        output_list_train = np.array(output_list_train)
        input_list_test = np.array(input_list_test)
        state_list_test = np.array(state_list_test)
        output_list_test = np.array(output_list_test)

        return input_list_train, state_list_train, output_list_train, input_list_test, state_list_test, \
               output_list_test, init_states_train, init_states_test

    def generate_data_2(self, batch_size, n_steps, noise_std=0.5):

        np.random.seed(11111)

        input_list_train = []
        output_list_train = []
        state_list_train = []
        input_list_test = []
        output_list_test = []
        state_list_test = []

        targets = np.meshgrid([0, -np.pi, np.pi], [0.], [0., -np.pi, np.pi], [0])
        targets = np.reshape(np.array(targets).T, [9, 1, 4])

        init_states_train = [np.pi*(np.random.rand(1, 4)*2 - 1) for _ in range(batch_size)]

        i = 0
        for batch in range(batch_size):
            target = targets[i]
            i = (i+1)%targets.shape[0]

            if batch < batch_size/2:
                states, inputs, outputs = self.closed_loop_sim(init_states_train[batch], steps=n_steps,
                                                               target=target, noise_std=noise_std)
                input_list_train.append(inputs)
                output_list_train.append(outputs)
                state_list_train.append(states)

            else:
                states, inputs, outputs = self.simulate_free(init_states_train[batch], steps=n_steps, noise_std=noise_std)

                input_list_train.append(inputs)
                output_list_train.append(outputs)
                state_list_train.append(states)

        init_states_test = [np.pi * (np.random.rand(1, 4) * 2 - 1) for _ in range(batch_size)]

        i = 0
        for batch in range(batch_size):
            target = targets[i]
            i = (i + 1) % targets.shape[0]

            if batch < batch_size / 2:
                states, inputs, outputs = self.closed_loop_sim(init_states_test[batch], steps=n_steps,
                                                               target=target, noise_std=noise_std)
                input_list_test.append(inputs)
                output_list_test.append(outputs)
                state_list_test.append(states)

            else:
                states, inputs, outputs = self.simulate_free(init_states_test[batch], steps=n_steps,
                                                             noise_std=noise_std)

                input_list_test.append(inputs)
                output_list_test.append(outputs)
                state_list_test.append(states)

        for i, state in enumerate(init_states_train):
            init_states_train[i] = state[0]

        for i, state in enumerate(init_states_test):
            init_states_test[i] = state[0]

        init_states_train = np.array(init_states_train)
        init_states_test = np.array(init_states_test)
        input_list_train = np.array(input_list_train)
        state_list_train = np.array(state_list_train)
        output_list_train = np.array(output_list_train)
        input_list_test = np.array(input_list_test)
        state_list_test = np.array(state_list_test)
        output_list_test = np.array(output_list_test)

        return input_list_train, state_list_train, output_list_train, input_list_test, state_list_test, \
               output_list_test, init_states_train, init_states_test

    def evaluate_performance(self, predictions, true_output):
        p_idx = []
        for i in range(self.n_outputs):
            p_idx.append(100. * (1. - np.sqrt(np.sum((np.square(predictions[:,:,i] - true_output[:,:,i])) / np.sum(np.square(np.mean(true_output[:,:,i]) - true_output[:,:,i]))))))
        return p_idx

    def visualize_results(self, predictions_train, output_train, predictions_test, output_test, network):
        time = np.arange(output_train.shape[1]) * self.Ts

        performance_idx = self.evaluate_performance(predictions_test, output_test)

        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        ax[0].set_title('Angle_1 predictions on training set')
        ax[0].plot(time, predictions_train[0,:,0], label='predicted')
        ax[0].plot(time, output_train[0,:,0], label='real')
        ax[0].legend()
        ax[0].set_ylabel('Angle [rad]')
        ax[1].set_title('Angle_2 predictions on training set')
        ax[1].plot(time, predictions_train[0, :, 1], label='predicted')
        ax[1].plot(time, output_train[0, :, 1], label='real')
        ax[1].legend()
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Angle [rad]')

        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        if network.observer_name is not None:
            ax[0].set_title('Angle_1 predictions on test set, {} network + {} filter, fit: {:.2f} %'.format(network.name,
                                                                                                 network.observer_name,
                                                                                                 performance_idx[0]))
            ax[1].set_title('Angle_2 predictions on test set, {} network + {} filter, fit: {:.2f} %'.format(network.name,
                                                                                                 network.observer_name,
                                                                                                 performance_idx[1]))
        else:
            ax[0].set_title('Angle_1 predictions on test set, {} network, fit: {:.2f} %'.format(network.name, performance_idx[0]))
            ax[1].set_title('Angle_2 predictions on test set, {} network, fit: {:.2f} %'.format(network.name, performance_idx[1]))

        ax[0].plot(time, predictions_test[0,:,0], label='predicted')
        ax[0].plot(time, output_test[0,:,0], label='real')
        ax[0].legend()
        ax[0].set_ylabel('Angle [rad]')
        ax[1].plot(time, predictions_test[0, :, 1], label='predicted')
        ax[1].plot(time, output_test[0, :, 1], label='real')
        ax[1].legend()
        ax[1].set_ylabel('Angle [rad]')
        ax[1].set_xlabel('Time [s]')


def main():
    system_to_test = 'tanks'

    n_steps = 500
    if system_to_test == 'tanks':
        model = Two_tank_model()
        x_0 = mat([-0.436, 0.5276, 13.746]).astype('float32')
        y_0 = mat([7.025])
        u_ = 15.6
        dist_ = 0.55
        u_series = np.random.randn(n_steps, 1) + u_
        dist_series = np.ones([n_steps, 1]) * dist_
        state_series, output_series = model.run(x_0, u_series, dist_series, n_steps)
        print(output_series.shape)
        print(state_series.shape)

    system_to_test = 'AUV'
    if system_to_test == 'AUV':
        x_0 = mat([0., 0., 0.])
        u_series = np.random.randn(n_steps, 2)
        state_series = AUV_model().run(x_0, u_series, 500)
        print(state_series.shape)


if __name__ == '__main__':
    main()
