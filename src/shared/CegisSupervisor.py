# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing

import torch

from src.shared.components.cegis import Cegis
from src.shared.consts import ActivationType, CegisStateKeys


def worker(cegis_config, id, result, run, base_seed=167):
    torch.manual_seed(base_seed + id)
    attempt = 0
    while run.is_set():
        attempt += 1
        c = Cegis(cegis_config)
        c.solve()
        res = c._result[0]["found"]
        # learn_method is unpickleable because of the dreal/z3 vars in
        # the certificate class. A better solution would be to write a
        # custom get state and set state for the learner / certificate
        # classes that ignores the learn_method attribute or symbolic
        # variables.
        c.learner.learn_method = None
        if res and run.is_set():
            run.clear()
            result["id"] = id
            result["attempt" + str(id)] = attempt
            result["res" + str(id)] = res
            result["cert" + str(id)] = c.learner
    return result


class CegisSupervisor:
    """Runs CEGIS in parallel and returns the first result found. Uses a shared dict"""

    def __init__(self, max_P=1):
        """Initializes the CegisSupervisor to run CEGIS in parallel.

        Args:
            max_P (int, optional): max processes to spawn. Defaults to 1.
        """

        self.max_processes = max_P

    def run(self, cegis_config):
        """Runs CEGIS in parallel and returns the first result found."""
        # check which cegis to run
        stop = False
        c = None
        procs = []
        return_dict = multiprocessing.Manager().dict()
        run = multiprocessing.Manager().Event()
        run.set()
        id = 0
        while not stop:
            while len(procs) < self.max_processes and not stop:
                # change config
                p = multiprocessing.Process(
                    target=worker, args=(cegis_config, id, return_dict, run)
                )
                # p.daemon = True
                p.start()
                id += 1
                procs.append(p)
            dead = [not p.is_alive() for p in procs]
            if not run.is_set() and any(dead):
                [p.terminate() for p in procs]
                return return_dict


def worker_Q(cegis_config, id, queue, run, base_seed=167):

    attempt = 0
    result = None
    torch.manual_seed(base_seed + id)
    while run.is_set():
        c = Cegis(cegis_config)
        c.solve()
        res = c._result[0]["found"]
        # result["res" + str(id)] = res
        # learn_method is unpickleable because of the dreal/z3 vars in
        # the certificate class. A better solution would be to write a
        # custom get state and set state for the learner / certificate
        # classes that ignores the learn_method attribute or symbolic
        # variables.
        c.learner.learn_method = None
        attempt += 1
        if res and run.is_set():
            # Add the id to the label as a sanity check (ensures returned result is from the correct process)
            run.clear()
            result = queue.get()
            result["id"] = id
            result["res" + str(id)] = res
            result["cert" + str(id)] = c.learner
            result["attempt" + str(id)] = attempt
            queue.put(result)
        if id == 0:
            cegis_config.ACTIVATION = [ActivationType.SQUARE]
    return result


class CegisSupervisorQ:
    """Runs CEGIS in parallel and returns the first result found. Uses a queue to communicate with the workers.

    I think a queue is better but I don't really know why. Possibly uses fewer processes and is safer"""

    def __init__(self, timeout_sec=1, max_P=1):
        self.cegis_timeout_sec = timeout_sec
        self.max_processes = max_P

    def run(self, cegis_config):
        stop = False
        c = None
        procs = []
        queue = multiprocessing.Manager().Queue()
        res = {}
        queue.put(res)
        run = multiprocessing.Manager().Event()
        base_seed = torch.initial_seed()
        run.set()
        id = 0
        while not stop:
            while len(procs) < self.max_processes and not stop:
                # change config
                p = multiprocessing.Process(
                    target=worker_Q, args=(cegis_config, id, queue, run, base_seed)
                )
                p.daemon = True
                p.start()
                id += 1
                procs.append(p)
            dead = [not p.is_alive() for p in procs]
            if any(dead) and not run.is_set():
                [p.terminate() for p in procs]
                res = queue.get()
                # queue.close()
                return res
