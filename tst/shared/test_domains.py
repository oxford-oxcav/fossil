# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import numpy as np
import torch
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

    def test_hyperbox_boundaries(self):
        two_dim = Rectangle([0, -1], [1, 2])
        three_dim = Rectangle([0, -1, 2], [1, 2, 3])
        torch.manual_seed(0)
        points_on_two_dim = two_dim.sample_border(10).detach().numpy()
        points_on_three_dim = three_dim.sample_border(10).detach().numpy()
        points_outside_two_dim = np.array([[-1, -1], [2, -1], [0, -2], [1, 3]])
        points_outside_three_dim = np.array(
            [
                [-1, -1, 2],
                [2, -1, 2],
                [0, -2, 2],
                [1, 3, 2],
                [-1, -1, 3],
                [2, -1, 3],
                [0, -2, 3],
                [1, 3, 3],
            ]
        )
        x2 = [z3.Real("x%d" % i) for i in range(2)]
        x3 = [z3.Real("x%d" % i) for i in range(3)]
        twod_sym = two_dim.generate_boundary(x2)
        threed_sym = three_dim.generate_boundary(x3)
        twod = z3.simplify(twod_sym)
        threed = z3.simplify(threed_sym)
        for p in points_on_two_dim:
            p = p.round(3)
            sub = [(x2[i], z3.RealVal(p[i])) for i in range(2)]
            self.assertTrue(z3.simplify(z3.substitute(twod, sub)))
        for p in points_on_three_dim:
            p = p.round(3)
            sub = [(x3[i], z3.RealVal(p[i])) for i in range(3)]
            self.assertTrue(z3.simplify(z3.substitute(threed, sub)))
        for p in points_outside_two_dim:
            sub = [(x2[i], z3.RealVal(p[i])) for i in range(2)]
            self.assertFalse(z3.simplify(z3.substitute(twod, sub)))
        for p in points_outside_three_dim:
            sub = [(x3[i], z3.RealVal(p[i])) for i in range(3)]
            self.assertFalse(z3.simplify(z3.substitute(threed, sub)))

    def test_hyperbox_boundaries_2(self):
        two_dim = Rectangle([0, -1], [1, 2])
        three_dim = Rectangle([0, -1, 2], [1, 2, 3])
        four_dim = Rectangle([0, -1, 2, 3], [1, 2, 3, 4])
        boxes = [two_dim, three_dim, four_dim]
        for box in boxes:
            x = [z3.Real("x%d" % i) for i in range(box.dimension)]
            border = box.generate_boundary(x)
            all = box.generate_domain(x)
            interior = box.generate_interior(x)
            border_alt = z3.And(all, z3.Not(interior))
            s = z3.Solver()
            # check if any point lies on the border and not in all minus interior
            s.add(border != border_alt)
            res = s.check()
            self.assertTrue(res == z3.unsat)


if __name__ == "__main__":
    unittest.main()
