from experiments.benchmarks.non_poly_0 import test_lnn as np0
from experiments.benchmarks.non_poly_1 import test_lnn as np1
from experiments.benchmarks.non_poly_2 import test_lnn as np2
from experiments.benchmarks.non_poly_3 import test_lnn as np3
from experiments.benchmarks.poly_1 import test_lnn as p1
from experiments.benchmarks.poly_2 import test_lnn as p2
from experiments.benchmarks.poly_3 import test_lnn as p3
from experiments.benchmarks.poly_4 import test_lnn as p4


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
