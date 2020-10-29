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
