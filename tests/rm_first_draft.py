from sympy import *
from tests.ReactiveModule import Atom, Cmd


(x0, x1, x2) = symbols('x0 x1 x2')
        
A1 = Atom(controls = [x0], reads = [x0], awaits = [],
          init = [
              Cmd(True, [0])
          ],
          flow = [
              Cmd(True, [-x0**3 + x1, -x1]),
          ],
          update = [])

A2 = Atom(controls = [x0], reads = [x0], awaits = [],
          init = [
              Cmd(True, [0])
          ],
          flow = [
              Cmd(x0**2 + x1**2 <= 5, [-x0**3 + x1, -x1]),
              Cmd(x0**2 + x1**2 > 5, [-x0 ** 3 + x1, -x2]),
          ],
          update = [])

