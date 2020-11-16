import torch
from experiments.benchmarks.barr_1 import main as b1_test
from experiments.benchmarks.hybrid_barrier import main as hybrid_test
from experiments.benchmarks.barr_3 import main as b3_test
from experiments.benchmarks.barr_2 import main as b2_test
from experiments.benchmarks.obstacle_avoidance import main as b4_test


def nips_benchmarks():
    b1_test()
    b2_test()
    b3_test()
    b4_test()
    hybrid_test()


if __name__ == '__main__':
    torch.manual_seed(167)
    nips_benchmarks()
