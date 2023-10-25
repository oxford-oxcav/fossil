# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import numpy as np
import z3

try:
    import dreal
except:
    print("No dreal")

from fossil.domains import Sphere, Rectangle


class DomainsTest(unittest.TestCase):
    def setUp(self):
        self.rectangle1 = Rectangle([-10, -5, 5], [5, 20, 6])
        self.rectangle2 = Rectangle([0, 0], [2, 3])
        self.sphere1 = Sphere([0, 0], 10)
        self.sphere2 = Sphere([-1, 5, 0], 4)

    def test_z3(self):
        x0, x1, x2 = [z3.Real("x%d" % i) for i in range(3)]
        z3_r1 = z3.And(-10 <= x0, -5 <= x1, 5 <= x2, x0 <= 5, x1 <= 20, x2 <= 6)
        z3_r2 = z3.And(0 <= x0, 0 <= x1, x0 <= 2, x1 <= 3)
        z3_s1 = z3.simplify(z3.And(x0**2 + x1**2 <= 100))
        z3_s2 = z3.simplify(z3.And((x0 + 1) ** 2 + (x1 - 5) ** 2 + x2**2 <= 16))

        gen_r1 = self.rectangle1.generate_domain([x0, x1, x2])
        gen_r2 = self.rectangle2.generate_domain([x0, x1])
        gen_s1 = self.sphere1.generate_domain([x0, x1])
        gen_s2 = self.sphere2.generate_domain([x0, x1, x2])

        self.assertTrue(z3_r1.eq(z3.simplify(gen_r1)))
        self.assertTrue(z3_r2.eq(z3.simplify(gen_r2)))
        self.assertTrue(z3_s1.eq(z3.simplify(gen_s1)))
        self.assertTrue(z3_s2.eq(z3.simplify(gen_s2)))

    def test_dreal(self):
        x0, x1, x2 = [dreal.Variable("x%d" % i) for i in range(3)]
        dreal_r1 = dreal.And(-10 <= x0, x0 <= 5, -5 <= x1, x1 <= 20, 5 <= x2, x2 <= 6)
        dreal_r2 = dreal.And(0 <= x0, x0 <= 2, 0 <= x1, x1 <= 3)
        dreal_s1 = dreal.And(x0**2 + x1**2 <= 100)
        dreal_s2 = dreal.And((x0 + 1) ** 2 + (x1 - 5) ** 2 + x2**2 <= 16)

        gen_r1 = self.rectangle1.generate_domain([x0, x1, x2])
        gen_r2 = self.rectangle2.generate_domain([x0, x1])
        gen_s1 = self.sphere1.generate_domain([x0, x1])
        gen_s2 = self.sphere2.generate_domain([x0, x1, x2])

        self.assertTrue(dreal_r1.EqualTo(gen_r1))
        self.assertTrue(dreal_r2.EqualTo(gen_r2))
        self.assertTrue(dreal_s1.EqualTo(gen_s1))
        self.assertTrue(dreal_s2.EqualTo(gen_s2))


if __name__ == "__main__":
    unittest.main()
