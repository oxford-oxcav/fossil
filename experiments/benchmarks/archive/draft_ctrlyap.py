import matplotlib.pyplot as plt
import torch

class Net(torch.nn.Module):
    def __init__(self, layers, activations, ctrler, f):
        super(Net, self).__init__()
        self.layers_dim = layers
        self.activations = activations
        self.f = f
        self.ctrler = ctrler

        self.layers = []
        for i in range(len(self.layers_dim)-1):
            self.layers += [torch.nn.Linear(self.layers_dim[i], self.layers_dim[i+1])]

        self.mod = []
        for i in range(len(self.layers)-1):
            self.mod += [self.layers[i], self.activations[i]]
        self.mod.append(self.layers[-1])

        self.model = torch.nn.Sequential(*self.mod)

    def forward(self, x):

        ctrl = self.ctrler(x)
        f_next = self.f(x) + 2.*ctrl

        x = self.model(x)

        return f_next, x

    def train(self, x):

        epochs = 100
        # create your optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        loss = torch.nn.LeakyReLU(1e-4)

        loss_history = torch.zeros((epochs))

        for _ in range(epochs):

            # in your training loop:
            optimizer.zero_grad()  # zero the gradient buffers
            f_n, out = self.forward(x)
            min_ = (loss(5.*out) + loss(f_n)).sum()
            loss_history[_] = min_
            min_.backward()
            optimizer.step()  # Does the update

        return loss_history


class Ctrler(torch.nn.Module):
    def __init__(self, layers, activations):
        super(Ctrler, self).__init__()
        self.layers_dim = layers
        self.activations = activations

        self.layers = []
        for i in range(len(self.layers_dim)-1):
            self.layers += [torch.nn.Linear(self.layers_dim[i], self.layers_dim[i+1])]

        self.mod = []
        for i in range(len(self.layers)-1):
            self.mod += [self.layers[i], self.activations[i]]
        self.mod.append(self.layers[-1])

        self.model = torch.nn.Sequential(*self.mod)

    def forward(self, x):

        return self.model(x)


inputs = 5
outputs = 4

# controler
torch.manual_seed(777)
ctrl_layers = [inputs, outputs]
ctrl_activations = [torch.nn.Identity]
ctrl = Ctrler(layers=ctrl_layers, activations=ctrl_activations)
cparams = list(ctrl.parameters())
print(len(cparams))
print(cparams)

layers = [inputs, outputs]
activations = [torch.nn.ReLU()]


def f(x):
    return x[:, :-1]**2 - x[:, :-1]


torch.manual_seed(167)
net = Net(layers, activations, ctrl, f)
params = list(net.parameters())
print(len(params))
print(params)
print(net.layers)

print(net.layers)


data = torch.rand((500, 5))
f_n, x = net.forward(data)

print('-'*80)
losses = net.train(data)
params = list(net.parameters())
print(len(params))
print(params)
print('-'*80)
cparams = list(ctrl.parameters())
print(len(cparams))
print(cparams)

plt.plot(losses.detach().numpy())

plt.show()



