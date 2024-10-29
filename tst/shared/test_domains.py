# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import numpy as np
import torch
import z3
from cvc5 import pythonic as cvpy

try:
    import dreal
except:
    print("No dreal")

import fossil.domains as domains


class DomainsTest(unittest.TestCase):
    def setUp(self):
        self.rectangle1 = domains.Rectangle([-10, -5, 5], [5, 20, 6])
        self.rectangle2 = domains.Rectangle([0, 0], [2, 3])
        self.sphere1 = domains.Sphere([0, 0], 10)
        self.sphere2 = domains.Sphere([-1, 5, 0], 4)

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
        two_dim = domains.Rectangle([0, -1], [1, 2])
        three_dim = domains.Rectangle([0, -1, 2], [1, 2, 3])
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
        two_dim = domains.Rectangle([0, -1], [1, 2])
        three_dim = domains.Rectangle([0, -1, 2], [1, 2, 3])
        four_dim = domains.Rectangle([0, -1, 2, 3], [1, 2, 3, 4])
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


class TestSMTCreation(unittest.TestCase):
    """This just checks that the symbolic domain functions create the right type of object"""

    def setUp(self) -> None:
        x_dr = tuple([dreal.Variable(f"x{i}") for i in range(3)])
        x_z3 = tuple([z3.Real(f"x{i}") for i in range(3)])
        x_cvc = tuple([cvpy.Real(f"x{i}") for i in range(3)])
        self.vars = {x_dr: dreal.Formula, x_z3: z3.BoolRef, x_cvc: cvpy.BoolRef}

    def check_domain(self, domain):
        for x, t in self.vars.items():
            self.assertIsInstance(domain.generate_domain(x), t)

    def check_boundary(self, domain):
        for x, t in self.vars.items():
            self.assertIsInstance(domain.generate_boundary(x), t)

    def check_interior(self, domain):
        for x, t in self.vars.items():
            self.assertIsInstance(domain.generate_interior(x), t)

    def test_Rectangle(self):
        R = domains.Rectangle([0, 0, 0], [1, 1, 1])
        self.check_domain(R)
        self.check_boundary(R)
        self.check_interior(R)

    def test_Sphere(self):
        S = domains.Sphere([0, 0, 0], 1)
        self.check_domain(S)
        self.check_boundary(S)
        self.check_interior(S)

    def test_Ellipsoid(self):
        E = domains.Ellipse([1, 1, 1], [0, 0, 0], 1)
        self.check_domain(E)
        self.check_boundary(E)
        self.check_interior(E)

        # def test_positive_orthant(self):
        P = domains.PositiveOrthantSphere([0, 0, 0], 1)
        self.check_domain(P)
        # self.check_boundary(P) # function doesn't exist
        # self.check_interior(P) # function doesn't exist

    def test_Torus(self):
        T = domains.Torus([0, 0, 0], 1, 0.1)
        self.check_domain(T)
        self.check_boundary(T)
        # self.check_interior(T) # function doesn't exist

    def test_Union(self):
        R = domains.Rectangle([0, 0, 0], [1, 1, 1])
        S = domains.Sphere([0, 0, 0], 1)
        U = domains.Union(R, S)
        self.check_domain(U)
        # self.check_boundary(U)
        # self.check_interior(U)

    def test_Intersection(self):
        R = domains.Rectangle([0, 0, 0], [1, 1, 1])
        S = domains.Sphere([0, 0, 0], 1)
        I = domains.Intersection(R, S)
        self.check_domain(I)
        # self.check_boundary(I)
        # self.check_interior(I)

    def test_Complement(self):
        R = domains.Rectangle([0, 0, 0], [1, 1, 1])
        C = domains.Complement(R)
        self.check_domain(C)
        # self.check_boundary(C) # function doesn't exist
        # self.check_interior(C)


if __name__ == "__main__":
    unittest.main()
