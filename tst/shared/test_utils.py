# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
import time

import sympy as sp
from z3 import *

from fossil.utils import Timer, timer, z3_to_string


class UtilsTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.x_sp, self.y_sp = sp.symbols("x y")

    def test_whenTimerStartAndStop_shouldUpdateTimeAndSetRepr(self):
        t = Timer()
        s = 3
        t.start()
        time.sleep(s)
        t.stop()
        rep = "total={}s,min={}s,max={}s,avg={}s, N={}".format(
            t.sum, t.min, t.max, t.avg, t.n_updates
        )
        self.assertAlmostEqual(s, t.max, delta=1)
        self.assertEqual(t.max, t.min)
        self.assertEqual(t.max, t.avg)
        self.assertEqual(rep, str(t))

    def test_whenTimerRepeatedlyCalled_shouldSetAvg(self):
        t = Timer()
        n = 3
        s = 2
        for i in range(n):
            t.start()
            time.sleep(s)
            t.stop()
        self.assertAlmostEqual(t.avg, s, delta=0.5)

    def test_whenTimerStartAndStopInDecorator_shouldUpdateTime(self):
        t = Timer()
        s = 2

        @timer(t)
        def test():
            time.sleep(s)

        test()
        self.assertAlmostEqual(t.max, s, delta=0.5)
        self.assertEqual(t.min, t.max)
        self.assertEqual(t.min, t.avg)

    def test_whenTimerRepeatedlyCalledInDecorator_shouldUpdateTime(self):
        t = Timer()
        s = 2
        n = 3

        @timer(t)
        def test(i):
            self.assertGreater(i, -1)
            time.sleep(s)

        for i in range(n):
            test(i)

        self.assertAlmostEqual(t.max, s, delta=0.5)
        self.assertAlmostEqual(t.min, t.max, delta=0.5)
        self.assertAlmostEqual(t.min, t.avg, delta=0.5)

    def test_whenZ3_to_string_shouldReturnStringRepresentation(self):
        x, y = Reals("x y")
        f = (
            2 * x**3
            + y
            + 4 * x * y
            + x * RealVal("102013931209828137410/312943712437280123908791423") * y
        )
        e = "2 * x ** 3 + y + 4 * x * y + x * 102013931209828137410/312943712437280123908791423 * y".replace(
            " ", ""
        )
        self.assertEqual(z3_to_string(f).replace(" ", ""), e)


if __name__ == "__main__":
    unittest.main()
