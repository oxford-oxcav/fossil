import torch
from experiments.benchmarks.darboux import main as darboux_test
from experiments.benchmarks.hybrid import main as hybrid_test
from experiments.benchmarks.obstacle_avoidance import main as oa_test
from experiments.benchmarks.elementary import main as elementary_test
from experiments.benchmarks.pj_mod import main as pjmod_test


def nips_benchmarks():
    darboux_test()
    hybrid_test()
    oa_test()
    elementary_test()
    pjmod_test()


if __name__ == '__main__':
    torch.manual_seed(167)
    nips_benchmarks()
