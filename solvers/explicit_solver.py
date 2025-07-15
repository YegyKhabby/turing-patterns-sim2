import numpy as np

def laplacian(Z, dx):
    """
    Second-order finite difference Laplacian (∇²U) on a 2D Cartesian grid.
    """
    """L = np.zeros_like(U)
    L[1:-1, 1:-1] = (
        U[2:, 1:-1] + U[:-2, 1:-1] + U[1:-1, 2:] + U[1:-1, :-2] - 4 * U[1:-1, 1:-1]
    ) / dx**2
    return L"""
    return (
        -4 * Z
        + np.roll(Z, (1, 0), (0, 1)) + np.roll(Z, (-1, 0), (0, 1))
        + np.roll(Z, (0, 1), (0, 1)) + np.roll(Z, (0, -1), (0, 1))
    ) / dx**2

def explicit_euler(u, v, f, g, D_u, D_v, dx, dt):
    """
    One explicit Euler step for u and v using arbitrary functions f(u,v), g(u,v).
    """
    Lu = laplacian(u, dx)
    Lv = laplacian(v, dx)

    u_new = u + dt * (D_u * Lu + f(u, v))
    v_new = v + dt * (D_v * Lv + g(u, v))

    return u_new, v_new
