# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import matplotlib.pyplot as plt
from fossil.domains import Ellipse, Sphere


class test_domain_plots(unittest.TestCase):
    def test_sphereAsEllipse2D(self):
        """
        check sphere as Ellipse 2D, written as
        ( x - 1)^2 +  (y + 7)^2 = 0.5^2
        """

        centre = [1.0, -7.0]
        coeffs = [1.0, 1]
        radius = 0.5
        e = Ellipse(coeffs=coeffs, centre=centre, radius=radius)

        s = Sphere(centre=centre, radius=radius)

        fig, ax = plt.subplots()
        e.plot(fig, ax)
        s.plot(fig, ax, label="unsafe")
        plt.grid()
        plt.legend()
        plt.title("Two figures overlapping")
        plt.show()

        self.assertTrue(True)

    def test_checkEllipses2D(self):
        """
        check ellipses 2D, written as
        ( 2x - 0.5)^2 + (0.5 y + 7)^2 = 0.5^2
        """

        centre = [0.5, -7.0]
        coeffs = [2.0, 0.5]
        radius = 0.5
        e = Ellipse(coeffs=coeffs, centre=centre, radius=radius)

        fig, ax = plt.subplots()
        e.plot(fig, ax)
        plt.grid()
        plt.title("x in [0.0, 0.5], y in [-13, -15]")
        plt.show()

        self.assertTrue(True)

    def test_generateData(self):
        """
        test data generation from
        ellipses 2D, written as
        ( 2x - 0.5)^2 + (0.5 y + 7)^2 = 0.5^2
        """

        centre = [0.5, -7.0]
        coeffs = [2.0, 0.5]
        radius = 0.5
        e = Ellipse(coeffs=coeffs, centre=centre, radius=radius)

        samples = e.generate_data(batch_size=1000)

        fig, ax = plt.subplots()
        e.plot(fig, ax)
        plt.grid()
        plt.title("x in [0.0, 0.5], y in [-13, -15]")

        plt.scatter(samples[:, 0], samples[:, 1], c="r")

        plt.show()

    def test_generateData_dimselect(self):
        """
        test data generation from
        ellipses 2D, written as
        ( 2x - 0.5)^2 + (0.5 y + 7)^2 + (2z + 6)^2 = 0.5^2
        """

        centre = [0.5, -7.0, -6.0]
        coeffs = [2.0, 0.5, 2.0]
        radius = 0.5
        ds = [0, 1]
        e = Ellipse(coeffs=coeffs, centre=centre, radius=radius, dim_select=ds)

        samples = e.generate_data(batch_size=1000)

        fig, ax = plt.subplots()
        e.plot(fig, ax)
        plt.grid()
        plt.title("x in [0.0, 0.5], y in [-13, -15]")

        plt.scatter(samples[:, 0], samples[:, 1], c="r")

        # Other axes
        ds = [1, 2]
        e = Ellipse(coeffs=coeffs, centre=centre, radius=radius, dim_select=ds)
        samples = e.generate_data(batch_size=1000)

        fig, ax = plt.subplots()
        e.plot(fig, ax)
        plt.grid()
        plt.title("x in [-13, -15], y in [-2.7, -3.25 ]")
        plt.scatter(samples[:, 0], samples[:, 1], c="r")

        plt.show()


if __name__ == "__main__":
    unittest.main()
