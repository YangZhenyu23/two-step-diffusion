import math
import unittest

import numpy as np

from two_step_kpp.data_io import periodic_l2_sq, wrap_torus


class TorusHelpersTest(unittest.TestCase):
    def test_wrap_torus(self):
        x = np.array([0.0, 2 * math.pi, 2 * math.pi + 0.1, -0.1])
        y = wrap_torus(x)
        self.assertTrue(np.all((y >= 0.0) & (y < 2 * math.pi)))
        self.assertAlmostEqual(y[1], 0.0)
        self.assertAlmostEqual(y[2], 0.1, places=5)

    def test_periodic_l2_sq_near_boundary(self):
        a = np.zeros((1, 1), dtype=np.float64)
        b = np.full((1, 1), 2 * math.pi - 0.01, dtype=np.float64)
        expected = 0.01**2
        self.assertAlmostEqual(float(periodic_l2_sq(a, b)), expected, places=5)


if __name__ == "__main__":
    unittest.main()
