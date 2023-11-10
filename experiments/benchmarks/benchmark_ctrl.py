import math

import torch

import experiments.benchmarks.models as models
import fossil.control as control
from fossil import certificate
from fossil.consts import *
from fossil.domains import *


def ctrllyap_identity(ctrler):
    outer = 10.0
    inner = 0.1
    batch_size = 5000
    open_loop = models.Identity()

    XD = Torus([0.0, 0.0], outer, inner)
    equilibrium = torch.zeros((1, 2))

    f = models.GeneralClosedLoopModel(open_loop, ctrler)

    domains = {
        certificate.XD: XD.generate_domain,
    }
    data = {
        certificate.XD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def ctrllyap_nonpolylyap(ctrler):
    outer = 10.0
    inner = 0.1
    batch_size = 1000
    open_loop = models.DTAhmadi()

    XD = Torus([0.0, 0.0], outer, inner)

    f = models.GeneralClosedLoopModel(open_loop, ctrler)

    domains = {
        certificate.XD: XD.generate_domain,
    }
    data = {
        certificate.XD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def ctrllyap_unstable(ctrler):
    outer = 10.0
    inner = 0.1
    batch_size = 5000
    open_loop = models.Benchmark2()

    XD = Torus([0.0, 0.0], outer, inner)
    equilibrium = torch.zeros((1, 2))

    f = models.GeneralClosedLoopModel(open_loop, ctrler)

    domains = {
        certificate.XD: XD.generate_domain,
    }
    data = {
        certificate.XD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def ctrllyap_lorenz_sys(ctrler):
    outer = 5.0
    inner = 0.1
    batch_size = 6000
    open_loop = models.LorenzSystem()

    XD = Torus([0.0, 0.0, 0.0], outer, inner)
    equilibrium = torch.zeros((1, 3))

    f = models.GeneralClosedLoopModel(open_loop, ctrler)

    domains = {
        certificate.XD: XD.generate_domain,
    }
    data = {
        certificate.XD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(3)


def ctrlbarr_car(ctrler):
    outer = 1
    batch_size = 1000
    open_loop = models.CtrlCar()

    XD = Torus([0.0, 0.0, 0.0], outer, 0.1)
    XI = Sphere([0.7, 0.7, 0.7], 0.2)
    XU = Sphere([-0.7, -0.7, -0.7], 0.2)
    XG = Sphere([0.3, 0.3, 0.3], 0.05)

    f = models.GeneralClosedLoopModel(open_loop, ctrler)

    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(3)


def linear_unstable_trajectory():
    outer = 2
    inner = 0.01
    batch_size = 1000
    open_loop = models.Benchmark1()

    XD = Torus([0.0, 0.0], outer, inner)
    equilibrium = torch.zeros((1, 2))

    ctrler = control.TrajectoryStable(
        inputs=2,
        outputs=2,
        layers=[4],
        activations=[ActivationType.LINEAR],
        time_domain=TimeDomain.CONTINUOUS,
        equilibrium=equilibrium,
        steps=50,
    )
    optim = torch.optim.AdamW(ctrler.parameters())
    ctrler.learn(XD.generate_data(batch_size), open_loop, optim)
    f = models.GeneralClosedLoopModel(open_loop, ctrler)

    domains = {
        certificate.XD: XD.generate_domain,
    }
    data = {
        certificate.XD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def general_linear_unstable_trajectory():
    outer = 2
    inner = 0.01
    batch_size = 1000
    open_loop = models.Benchmark2()

    XD = Torus([0.0, 0.0], outer, inner)
    equilibrium = torch.zeros((1, 2))

    ctrler = control.TrajectoryStable(
        inputs=2,
        outputs=3,
        layers=[4],
        activations=[ActivationType.LINEAR],
        time_domain=TimeDomain.CONTINUOUS,
        equilibrium=equilibrium,
        steps=50,
    )
    optim = torch.optim.AdamW(ctrler.parameters())
    ctrler.learn(XD.generate_data(batch_size), open_loop, optim)
    f = models.GeneralClosedLoopModel(open_loop, ctrler)

    domains = {
        certificate.XD: XD.generate_domain,
    }
    data = {
        certificate.XD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def ctrllyap_linear_dt(ctrler):
    outer = 2
    inner = 0.01
    batch_size = 1000
    open_loop = models.BenchmarkDT1()

    XD = Torus([0.0, 0.0], outer, inner)

    f = models.GeneralClosedLoopModel(open_loop, ctrler)

    domains = {
        certificate.XD: XD.generate_domain,
    }
    data = {
        certificate.XD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def car_traj_control():
    outer = 1
    batch_size = 1000
    open_loop = models.Car()

    XD = Torus([0.0, 0.0, 0.0], outer, 0.1)
    XI = Sphere([0.7, 0.7, 0.7], 0.2)
    XU = Sphere([-0.7, -0.7, -0.7], 0.2)
    XG = Sphere([0.3, 0.3, 0.3], 0.05)

    ctrler = control.TrajectorySafeStableCT(
        dim=3,
        layers=[3],
        activations=[ActivationType.LINEAR],
        time_domain=TimeDomain.CONTINUOUS,
        goal=XG,
        unsafe=XU,
        steps=20,
    )
    optim = torch.optim.AdamW(ctrler.parameters())
    ctrler.learn(XD.generate_data(batch_size), open_loop, optim)
    f = models._PreTrainedModel(open_loop, ctrler)

    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(3)


def inv_pendulum_ctrl():
    outer = 1
    inner = 0.1
    batch_size = 1500
    open_loop = models.SineModel()

    XD = Torus([0.0, 0.0], outer, inner)
    equilibrium = torch.zeros((1, 2))

    ctrler = control.TrajectoryStable(
        inputs=2,
        outputs=2,
        layers=[5],
        activations=[ActivationType.LINEAR],
        time_domain=TimeDomain.CONTINUOUS,
        equilibrium=equilibrium,
        steps=10,
    )
    optim = torch.optim.AdamW(ctrler.parameters())
    ctrler.learn(XD.generate_data(batch_size), open_loop, optim)
    f = models.GeneralClosedLoopModel(open_loop, ctrler)

    domains = {
        certificate.XD: XD.generate_domain,
    }
    data = {
        certificate.XD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def linear_satellite(ctrler):
    batch_size = 1500
    ins = 6

    open_loop = models.LinearSatellite()

    lowers = [-2.0, -2.0, -2.0, -1.0, -1.0, -1.0]
    uppers = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]

    XD = Rectangle(lb=lowers, ub=uppers)

    XI = Torus(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        outer_radius=1.0,
        inner_radius=0.75,
        dim_select=[0, 1, 2],
    )

    class Unsafe(Set):
        def __init__(self):
            self.inner_radius = 0.25
            self.outer_radius = 1.5

        def generate_domain(self, v):
            x1, x2, x3, x4, x5, x6 = v
            f = self.set_functions(v)
            _And = f["And"]
            _Or = f["Or"]
            return _Or(
                x1**2 + x2**2 + x3**2 <= self.inner_radius**2,
                x1**2 + x2**2 + x3**2 >= self.outer_radius**2,
            )

        def generate_data(self, batch_size):
            n0 = int(batch_size / 2)
            n1 = batch_size - n0
            # there is no method to generate data OUTSIDE a sphere,
            # so we generate data in a rectangle and check if they are outside a sphere
            # not the nicest solution, but it works
            outsphere = torch.zeros((n1, 6))
            lb, ub = lowers, uppers
            dom = square_init_data([lb, ub], 10 * n1)
            k = 0
            # check if outside a sphere
            for idx in range(10 * n1):
                sample = dom[idx, :]
                if (
                    sample[0] ** 2 + sample[1] ** 2 + sample[2] ** 2
                    >= self.outer_radius**2
                ):
                    outsphere[k, :] = sample
                    k += 1
                if k >= n1:
                    break

            return torch.cat(
                [
                    Sphere(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        radius=self.inner_radius,
                        dim_select=[0, 1, 2],
                    ).generate_data(n0),
                    outsphere,
                ]
            )

        def check_containment(self, x: torch.Tensor) -> torch.Tensor:
            return torch.logical_or(
                (x).norm(2, dim=-1) <= self.inner_radius,
                (x).norm(2, dim=-1) <= self.outer_radius,
            )

        def check_containment_grad(self, x: torch.Tensor) -> torch.Tensor:
            # check containment and return a tensor with gradient
            # returns 0 if it IS contained, a positive number otherwise
            return torch.relu((x).norm(2, dim=-1) - self.inner_radius) + torch.relu(
                self.outer_radius - x.norm(2, dim=-1)
            )

    XU = Unsafe()

    XG = Torus(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        outer_radius=0.9,
        inner_radius=0.65,
        dim_select=[0, 1, 2],
    )

    f = models.GeneralClosedLoopModel(open_loop, ctrler)

    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(ins)
