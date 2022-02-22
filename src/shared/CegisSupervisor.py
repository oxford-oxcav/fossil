# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing

from src.shared.components.cegis import Cegis


class CegisSupervisor:
    def __init__(self):
        self.cegis_timeout_sec = 10

    def run(self):
        # check which cegis to run
        stop = False
        c = None
        cegis_config = {}
        while not stop:
            # change config
            c = Cegis(**cegis_config)
            p = multiprocessing.Process(target=c.solve)
            p.start()
            p.join(self.cegis_timeout_sec)
            stop = not p.is_alive()
            p.terminate()
        print("CEGIS output")
        print(c.result)


if __name__ == "__main__":
    main()
