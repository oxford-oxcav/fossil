class Component:
    def __init__(self):
        self._to_next_component = lambda x: x

    def get(self, **kw):
        raise NotImplemented("Not implemented in " + self.__class__.__name__)

    def to_next_component(self, fn):
        self._to_next_component = fn
