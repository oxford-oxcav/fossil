# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
from experiments.benchmarks.lyap.non_poly_0 import test_lnn as np0
from experiments.benchmarks.lyap.non_poly_1 import test_lnn as np1
from experiments.benchmarks.lyap.non_poly_2 import test_lnn as np2
from experiments.benchmarks.lyap.non_poly_3 import test_lnn as np3
from experiments.benchmarks.lyap.poly_1 import test_lnn as p1
from experiments.benchmarks.lyap.poly_2 import test_lnn as p2
from experiments.benchmarks.lyap.poly_3 import test_lnn as p3
from experiments.benchmarks.lyap.poly_4 import test_lnn as p4


def lyap_benchmarks():
    np0()
    np1()
    np2()
    np3()
    p1()
    p2()
    p3()
    p4()


if __name__ == '__main__':
    lyap_benchmarks()
