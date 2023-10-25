# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import time
from queue import Empty

import torch

from fossil import cegis


def worker_Q(cegis_config, id, queue, base_seed=0):
    seed = base_seed + id
    torch.manual_seed(seed)
    # np.random.seed(seed)
    c = cegis.Cegis(cegis_config)
    result = c.solve()
    print(result)
    # Remove the functions & symbolic vars from the result to avoid pickling errors
    if isinstance(result.cert, tuple):
        result.cert[0].clean()
        result.cert[1].clean()
    else:
        result.cert.clean()
    result.f.clean()
    success = result.res
    if success:
        logging.debug("Worker", id, "succeeded")
    elif not result.res:
        logging.debug("Worker", id, "failed")
    result_dict = {}
    # Add the id to the label as a sanity check (ensures returned result is from the correct process)
    result_dict["id"] = id
    result_dict["success"] = success
    result_dict["result" + str(id)] = result
    queue.put(result_dict)
    return result_dict


class CegisSupervisorQ:
    """Runs CEGIS in parallel and returns the first result found. Uses a queue to communicate with the workers."""

    def __init__(self, timeout_sec=1800, max_P=1):
        self.cegis_timeout_sec = timeout_sec
        self.max_processes = max_P

    def solve(self, cegis_config) -> cegis.Result:
        stop = False
        procs = []
        queue = multiprocessing.Manager().Queue()
        base_seed = torch.initial_seed()
        id = 0
        n_res = 0
        start = time.perf_counter()
        while not stop:
            while len(procs) < self.max_processes and not stop:
                p = multiprocessing.Process(
                    target=worker_Q, args=(cegis_config, id, queue, base_seed)
                )
                p.start()
                id += 1
                procs.append(p)
            dead = [not p.is_alive() for p in procs]

            try:
                res = queue.get(block=False)
                if res["success"]:
                    logging.debug("Success: Worker", res["id"])
                    [p.terminate() for p in procs]
                    _id = res["id"]
                    result = res["result" + str(_id)]
                    return result
                else:
                    n_res += 1
                if n_res == self.max_processes:
                    logging.debug("All workers failed, returning last result")
                    # Return the last result
                    _id = res["id"]
                    result = res["result" + str(_id)]
                    return result

            except Empty:
                pass

            # if time.perf_counter() - start > self.cegis_timeout_sec:
            #     # If the timeout has been reached then kill all processes and return
            #     logging.info("Timeout reached")
            #     [p.terminate() for p in procs]
            #     res = {}
            #     res["id"] = ""
            #     delta_t = time.perf_counter() - start
            #     res["success"] = False
            #     res["result"] = None
            #     return res
