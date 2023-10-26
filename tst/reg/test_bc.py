# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from experiments.benchmarks.barrier.barr_1 import main as b1_test
from experiments.benchmarks.barrier.hybrid_barrier import main as hybrid_test
from experiments.benchmarks.barrier.barr_3 import main as b3_test
from experiments.benchmarks.barrier.barr_2 import main as b2_test
from experiments.benchmarks.barrier.barr_4 import test_lnn as b4_test


def nips_benchmarks():
    b1_test()
    b2_test()
    b3_test()
    b4_test()
    hybrid_test()


if __name__ == "__main__":
    torch.manual_seed(167)
    nips_benchmarks()
