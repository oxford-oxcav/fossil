from sympy import *

class Cmd:

    def __init__(self, guard, output):
        self.guard = guard;
        self.output = output;

class Atom:

    def init(self, controls: list, reads: list, awaits: list,
             init: list, flow: list, update: list):
        assert all([isinstance(c, Cmd) for c in init])
        assert all([isinstance(c, Cmd) for c in flow])
        assert all([isinstance(c, Cmd) for c in update])
        
        self._controls = controls;
        self._reads = reads;
        self._awaits = awaits;
        self._init = init;
        self._flow = flow;
        self._update = update;

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            self.init(**kwargs)
        elif len(args) == 1:
            self.copyRename(*args, **kwargs)
        else:
            assert false
    
    def copyRename(self, atom: Atom, controls: list, reads: list, awaits: list):
        print("copy and rename")


(x,y,z) = symbols('x y z')
        
A1 = Atom(controls = [x], reads = [y, z], awaits = [],
          init = [
              Cmd(True, [0])
          ],
          flow = [
              Cmd(x <= 0, [x + z]),
              Cmd(x >= 0, [-2*x + y])
          ],
          update = [])

# A2 = Atom(A1, controls = [y], reads = [x, z], awaits = [])


class Module:

    def __init__(self, interface, external, private, atoms):
        print("init")
