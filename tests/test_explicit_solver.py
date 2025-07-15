import unittest
import numpy as np
from solvers.explicit_solver import explicit_euler

class TestExplicitSolver(unittest.TestCase):

    def test_shapes_and_finiteness(self):
        N = 50
        dx = 1.0 / N
        dt = 0.001
        D_u, D_v = 0.1, 0.05

        # Initial conditions
        u = np.ones((N, N)) + 0.01 * np.random.randn(N, N)
        v = np.zeros((N, N)) + 0.01 * np.random.randn(N, N)

        # Define arbitrary test functions
        def f(u, v): return -u * v**2 + 0.04 * (1 - u)
        def g(u, v): return u * v**2 - (0.06 + 0.04) * v

        u_new, v_new = explicit_euler(u, v, f, g, D_u, D_v, dx, dt)

        self.assertEqual(u.shape, u_new.shape)
        self.assertEqual(v.shape, v_new.shape)
        self.assertTrue(np.all(np.isfinite(u_new)))
        self.assertTrue(np.all(np.isfinite(v_new)))

if __name__ == '__main__':
    unittest.main()
