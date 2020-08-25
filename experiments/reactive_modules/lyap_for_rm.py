from experiments.benchmarks.benchmarks_lyap import *
from experiments.synth_rm_lyap import synthesis_rm_lyap
import sympy as sp
from experiments.ReactiveModule import Atom, Cmd

(x0, x1, x2) = sp.symbols('x0 x1 x2')

A1 = Atom(controls=[x0], reads=[x0], awaits=[],
          init=[
              Cmd(True, [0])
          ],
          flow=[
              Cmd(True, [-x0 ** 3 + x1, -x1]),
          ],
          update=[])

A2 = Atom(controls=[x0], reads=[x0], awaits=[],
          init=[
              Cmd(True, [0])
          ],
          flow=[
              Cmd(x0 ** 2 + x1 ** 2 - 5, [-x0 ** 3 + x1, -x1]),
              Cmd(x0 ** 2 + x1 ** 2 - 5, [-x0 ** 3 + x1, +x1]),
          ],
          update=[])


def main():

    input_rm = A1
    synthesis_rm_lyap(input_rm)


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
