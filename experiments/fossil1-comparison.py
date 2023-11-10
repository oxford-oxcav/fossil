# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import timeit
import os.path

import pandas as pd

from experiments.benchmarks import models
from fossil import analysis
from fossil import domains
from fossil import certificate
from fossil import cegis
from fossil.consts import *

### FOSSIL 1 Results

np1_stats = analysis.Stats(0.21, 0.48, 0.04, 1.58)
np1_succ = 10
p2_stats = analysis.Stats(11.71, 22.62, 0.35, 70.39)
p2_succ = 9
b1_stats = analysis.Stats(100.17, 0, 100.17, 100.17)
b1_succ = 1
b3_stats = analysis.Stats(101.72, 134.82, 16.80, 334.79)
b3_succ = 5

###

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Change this line for different works
RESULTS_DIR = BASE_DIR + "/results/"
RAW_RESULTS_DIR = RESULTS_DIR + "raw_fossil1_comparison.csv"


class Barr3Init(domains.Set):
    dimension = 2

    def generate_domain(self, v):
        x, y = v
        f = self.set_functions(v)
        _Or = f["Or"]
        _And = f["And"]
        return _Or(
            _And((x - 1.5) ** 2 + y**2 <= 0.25),
            _And(x >= -1.8, x <= -1.2, y >= -0.1, y <= 0.1),
            _And(x >= -1.4, x <= -1.2, y >= -0.5, y <= 0.1),
        )

    def generate_data(self, batch_size):
        n0 = int(batch_size / 3)
        n1 = n0
        n2 = batch_size - (n0 + n1)
        return torch.cat(
            [
                domains.circle_init_data((1.5, 0.0), 0.25, n0),
                domains.square_init_data([[-1.8, -0.1], [-1.2, 0.1]], n1),
                domains.add_corners_2d([[-1.8, -0.1], [-1.2, 0.1]]),
                domains.square_init_data([[-1.4, -0.5], [-1.2, 0.1]], n2),
                domains.add_corners_2d([[-1.4, -0.5], [-1.2, 0.1]]),
            ]
        )


class Barr3Unsafe(domains.Set):
    dimension = 2

    def generate_domain(self, v):
        x, y = v
        f = self.set_functions(v)
        _Or = f["Or"]
        _And = f["And"]
        return _Or(
            (x + 1) ** 2 + (y + 1) ** 2 <= 0.16,
            _And(0.4 <= x, x <= 0.6, 0.1 <= y, y <= 0.5),
            _And(0.4 <= x, x <= 0.8, 0.1 <= y, y <= 0.3),
        )

    def generate_data(self, batch_size):
        n0 = int(batch_size / 3)
        n1 = n0
        n2 = batch_size - (n0 + n1)
        return torch.cat(
            [
                domains.circle_init_data((-1.0, -1.0), 0.16, n0),
                domains.square_init_data([[0.4, 0.1], [0.6, 0.5]], n1),
                domains.add_corners_2d([[0.4, 0.1], [0.6, 0.5]]),
                domains.square_init_data([[0.4, 0.1], [0.8, 0.3]], n2),
                domains.add_corners_2d([[0.4, 0.1], [0.8, 0.3]]),
            ]
        )


class Barr1Unsafe(domains.Set):
    dimension = 2

    def generate_domain(self, v):
        x, y = v
        return x + y**2 <= 0

    def generate_data(self, batch_size):
        points = []
        limits = [[-2, -2], [0, 2]]
        while len(points) < batch_size:
            dom = domains.square_init_data(limits, batch_size)
            idx = torch.nonzero(dom[:, 0] + dom[:, 1] ** 2 <= 0)
            points += dom[idx][:, 0, :]
        return torch.stack(points[:batch_size])


def run_cegis(opts):
    c = cegis.Cegis(opts)
    start = timeit.default_timer()
    res = c.solve()
    stop = timeit.default_timer()
    res_file = RAW_RESULTS_DIR
    rec = analysis.Recorder(analysis.AnalysisConfig(results_file=res_file))
    rec.record(opts, res, stop - start)


def non_poly1():
    system = models.NonPoly0
    X = domains.Torus([0, 0], 1, 0.01)
    domain = {certificate.XD: X}
    data = {certificate.XD: X._generate_data(1000)}

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [5] * len(activations)

    ###
    #
    ###
    opts_nonpoly1 = CegisConfig(
        SYSTEM=system,
        DOMAINS=domain,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=CertificateType.LYAPUNOV,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        LLO=True,
        CEGIS_MAX_ITERS=25,
        VERBOSE=False,
    )
    run_cegis(opts_nonpoly1)


def poly2():
    system = models.Poly2
    X = domains.Torus([0, 0], 5, 0.01)
    domain = {certificate.XD: X}
    data = {certificate.XD: X._generate_data(1000)}
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [5] * len(activations)
    opts_poly2 = CegisConfig(
        SYSTEM=system,
        DOMAINS=domain,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=CertificateType.LYAPUNOV,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        LLO=True,
        CEGIS_MAX_ITERS=25,
        VERBOSE=False,
    )
    run_cegis(opts_poly2)


def barr_1():
    system = models.Barr1
    activations = [ActivationType.SIGMOID]
    XD = domains.Rectangle([-2, -2], [2, 2])
    XI = domains.Rectangle([0, 1], [1, 2])
    XU = Barr1Unsafe()
    domain = {certificate.XD: XD, certificate.XU: XU, certificate.XI: XI}
    data = {
        certificate.XD: XD._generate_data(500),
        certificate.XI: XI._generate_data(500),
        certificate.XU: XU._generate_data(500),
    }
    n_hidden_neurons = [5] * len(activations)
    opts_barr_1 = CegisConfig(
        SYSTEM=system,
        DOMAINS=domain,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=25,
        VERBOSE=0,
        SYMMETRIC_BELT=True,
    )
    run_cegis(opts_barr_1)


def barr_3():
    system = models.Barr3
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    XD = domains.Rectangle([-3, -2], [2.5, 1])
    XI = domains.Union(
        domains.Sphere([1.5, 0], 0.5),
        domains.Union(
            domains.Rectangle([-1.8, -0.1], [-1.2, 0.1]),
            domains.Rectangle([-1.4, -0.5], [-1.2, 0.1]),
        ),
    )

    XU = domains.Union(
        domains.Sphere([-1, -1], 0.4),
        domains.Union(
            domains.Rectangle([0.4, 0.1], [0.6, 0.5]),
            domains.Rectangle([0.4, 0.1], [0.8, 0.3]),
        ),
    )

    domain = {certificate.XD: XD, certificate.XU: XU, certificate.XI: XI}
    data = {
        certificate.XD: XD._generate_data(1000),
        certificate.XI: XI._generate_data(400),
        certificate.XU: XU._generate_data(400),
    }

    n_hidden_neurons = [10] * len(activations)
    opts_barr_3 = CegisConfig(
        SYSTEM=system,
        DOMAINS=domain,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=25,
        SYMMETRIC_BELT=False,
        VERBOSE=False,
    )
    run_cegis(opts_barr_3)


def make_table():
    # This function is so ugly
    output_file = RESULTS_DIR + "fossil_comparison"

    benchmarks_names = ["NonPoly0", "Poly2", "Barr1", "Barr3"]
    fossil1_res = {
        "Benchmark": benchmarks_names,
        "Success": [np1_succ, p2_succ, b1_succ, b3_succ],
        "avg_time": [np1_stats.mean, p2_stats.mean, b1_stats.mean, b3_stats.mean],
        "min_time": [np1_stats.min, p2_stats.min, b1_stats.min, b3_stats.min],
        "max_time": [np1_stats.max, p2_stats.max, b1_stats.max, b3_stats.max],
    }
    f1_df = pd.DataFrame(fossil1_res)

    f2_df = pd.read_csv(RAW_RESULTS_DIR)

    f2_df.rename(
        {"N_s": "$N_s$", "N_u": "$N_u$", "Benchmark_file": "Benchmark"},
        axis=1,
        inplace=True,
    )
    f2_df["Benchmark"].replace({"_": "\_"}, inplace=True, regex=True)
    f2_df["Activations"].replace({"_": "\_"}, inplace=True, regex=True)

    # grouped = df.groupby(["Benchmark_file"])
    vals = [
        "Total_Time",
        "Result",
    ]
    ind = [
        "Benchmark",
        "$N_s$",
        "Certificate",
        "Neurons",
        "Activations",
    ]
    table = pd.pivot_table(
        f2_df,
        values=vals,
        index=ind,
        aggfunc={"Total_Time": ["min", "mean", "max"], "Result": analysis.ratio},
    )
    table = table.loc[
        benchmarks_names,
        :,
    ]

    table.columns = table.columns.to_flat_index().str.join("_")
    table = table.reindex(
        columns=["Total_Time_min", "Total_Time_mean", "Total_Time_max", "Result_ratio"]
    )
    table["F1_min"] = f1_df.iloc[:, 3].values
    table["F1_mean"] = f1_df.iloc[:, 2].values
    table["F1_max"] = f1_df.iloc[:, 4].values
    table["F1_Success"] = 100 * f1_df.iloc[:, 1].values / 10  # 10 runs for fossil 1
    table.rename(
        {
            "Result_ratio": "F2_Success",
            "Total_Time_min": "F2_min",
            "Total_Time_mean": "F2_mean",
            "Total_Time_max": "F2_max",
        },
        axis=1,
        inplace=True,
    )
    table = pd.concat(
        [table.filter(like="F1"), table.filter(like="F2")],
        axis=1,
        keys=("Fosill 1.0", "Fossil 2.0"),
    )
    table.rename(
        {
            "F1_Success": "$S$",
            "F1_min": "$\min$",
            "F1_mean": "$\mu$",
            "F1_max": "$\max$",
            "F2_Success": "$S$",
            "F2_min": "$\min$",
            "F2_mean": "$\mu$",
            "F2_max": "$\max$",
        },
        axis=1,
        inplace=True,
    )
    pd.options.display.float_format = "{:,.2f}".format
    print(table)
    table.to_latex(
        output_file + ".tex",
        float_format="%.2f",
        bold_rows=False,
        escape=False,
        multicolumn_format="c",
    )

    table.columns = table.columns.map(" ".join)
    table.to_csv(output_file + ".csv", float_format="%.2f")
    # To flip multi indexing:
    # table.columns = table.columns.swaplevel(0,1)
    # table.sort_index(axis=1, level=0, inplace=True)
    # a.table_main()


if __name__ == "__main__":
    BASE_SEED = 169
    benchmarks = [non_poly1, poly2, barr_1, barr_3]
    N_REPEATS = 10
    for benchmark in benchmarks:
        for i in range(N_REPEATS):
            torch.manual_seed(BASE_SEED + i)
            benchmark()

    make_table()
