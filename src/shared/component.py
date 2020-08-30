class Component:
    def __init__(self):
        pass

    def get(self, **kw):
        raise NotImplemented("Not implemented in " + self.__class__.__name__)

    def to_next_component(self, out, component, **kw):
        return out
