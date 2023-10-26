# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


class Component:
    def __init__(self):
        pass

    def get(self, **kw):
        raise NotImplemented("Not implemented in " + self.__class__.__name__)

    def to_next_component(self, out, component, **kw):
        return out
