import torch
import timeit
from src.lyap.cegis_lyap import Cegis
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig
from src.shared.consts import VerifierType, LearnerType
from src.shared.cegis_values import CegisConfig
from src.plots.plot_lyap import plot_lyce
from functools import partial

from experiments.benchmarks.benchmark_3 import test_lnn as bench3
from experiments.benchmarks.non_poly_0 import test_lnn as np0
from experiments.benchmarks.non_poly_1 import test_lnn as np1
from experiments.benchmarks.non_poly_2 import test_lnn as np2
from experiments.benchmarks.non_poly_3 import test_lnn as np3


def lyap_benchmarks():
    bench3()
    np0()
    np1()
    np2()
    np3()


if __name__ == '__main__':
    lyap_benchmarks()
