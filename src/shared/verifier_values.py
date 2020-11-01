from aenum import Enum, NoAlias


class VerifierConfig(Enum, settings=NoAlias):
    DREAL_INFINITY              = 1e300
    # dReal will return infinity when:
    # - no counterexamples are found
    # - a smaller counterexample also exists
    # check again for a counterexample with the bound below
    DREAL_SECOND_CHANCE_BOUND   = 1e3

    @property
    def k(self):
        return self.name

    @property
    def v(self):
        return self.value
