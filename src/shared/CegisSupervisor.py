import multiprocessing

from src.barrier.cegis_barrier import Cegis as CegisBarrier
from src.lyap.cegis_lyap import Cegis as CegisLyapunov


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
            c = CegisBarrier(**cegis_config)
            p = multiprocessing.Process(target=c.solve)
            p.start()
            p.join(self.cegis_timeout_sec)
            stop = not p.is_alive()
            p.terminate()
        print("CEGIS output")
        print(c.result)




if __name__ == '__main__':
    main()
