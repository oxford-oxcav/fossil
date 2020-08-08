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

    def get_flow(self):
        return self._flow

    def get_guards(self):
        guards = []
        for f in self._flow:
            guards += [ f.guard ]
        return guards

    def get_vars(self):
        for f in self._flow:
            vars = set.union(*[set(expr.free_symbols) for expr in f.output])
        return vars

    def get_dynamics(self):
        dyna = []
        dyna += [self._flow[idx].output for idx in range(len(self._flow))]
        return dyna


class Module:

    def __init__(self, interface, external, private, atoms):
        print("init")
