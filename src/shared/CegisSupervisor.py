# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing

from src.shared.components.cegis import Cegis


class CegisSupervisor:
    def __init__(self, timeout_sec=1, max_P=1):
        self.cegis_timeout_sec = timeout_sec
        self.max_processes = max_P

    def run(self, cegis_config):
        # check which cegis to run
        stop = False
        c = None
        procs = []
        while len(procs) < self.max_processes and not stop:
            # change config
            c = Cegis(**cegis_config)
            p = multiprocessing.Process(target=c.solve)
            p.start()
            procs.append(p)
        return c.result

