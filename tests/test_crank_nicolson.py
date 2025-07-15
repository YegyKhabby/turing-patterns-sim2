import unittest
import numpy as np

from solvers.crank_nicolson import crank_nicolson
from models.gray_scott import gray_scott_f, gray_scott_g
from models.initialize import initialize

class TestCrankNicolson(unittest.TestCase):

    def setUp(self):
        self.N = 32
        self.dx = 1.0 / self.N
        self.dt = 1.0
        self.steps = 10
        self.Du = 0.16
        self.Dv = 0.08
        self.params = [0.04, 0.06]
        self.u0, self.v0 = initialize(self.N, pattern='gradient')

    def test_output_shape(self):
        u, v = crank_nicolson(self.u0, self.v0, self.Du, self.Dv, gray_scott_f, gray_scott_g, self.dt, self.dx, self.steps, self.params)
        self.assertEqual(u.shape, (self.N, self.N))
        self.assertEqual(v.shape, (self.N, self.N))

    def test_output_finite(self):
        u, v = crank_nicolson(self.u0, self.v0, self.Du, self.Dv, gray_scott_f, gray_scott_g, self.dt, self.dx, self.steps, self.params)
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertTrue(np.all(np.isfinite(v)))

    def test_reproducibility(self):
        u1, v1 = crank_nicolson(self.u0.copy(), self.v0.copy(), self.Du, self.Dv, gray_scott_f, gray_scott_g, self.dt, self.dx, self.steps, self.params)
        u2, v2 = crank_nicolson(self.u0.copy(), self.v0.copy(), self.Du, self.Dv, gray_scott_f, gray_scott_g, self.dt, self.dx, self.steps, self.params)
        np.testing.assert_allclose(u1, u2)
        np.testing.assert_allclose(v1, v2)

if __name__ == '__main__':
    unittest.main()
