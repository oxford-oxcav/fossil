from src.shared.component import Component


class Learner(Component):
    def __init__(self):
        super().__init__()

    def get(self, **kw):
        return self.learn(**kw)

    def learn(self, *args, **kwargs):
        return NotImplemented("Not implemented in " + self.__class__.__name__)
