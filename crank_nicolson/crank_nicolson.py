import numpy as np
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import factorized
import time

def laplacian_matrix(N, dx):
    """ 
    Creates a 2D Laplacian matrix with periodic boundary conditions

    args:
    - N: number of grid points in one spatial dimension
    - dx: grid spacing
    """
    e = np.ones(N)
    L = diags([e, -2*e, e], [-1, 0, 1], shape=(N, N)).tolil() # Construct tridiagonal matrix (Laplacian in 1D)
    L[0, -1] = L[-1, 0] = 1  # Periodic BCs
    return L / dx**2

# Combined Crank-Nicolson solver for general f and g with parameters
def crank_nicolson(u, v, Du, Dv, f, g, dt, dx, steps, p):
    """
    Crank-Nicolson solver
    
    Args:
    - u, v: initial 2D arrays (concentration fields)
    - Du, Dv: diffusion coefficients
    - f, g: functions defining the reaction kinetics; must take (u, v, p) as input
    - dt, dx: time and space step sizes
    - steps: number of time steps to simulate
    - p: a list or array of parameters passed to f and g
    """
    N = u.shape[0]
    I = identity(N)
    L1D = laplacian_matrix(N, dx)
    L2D = kron(I, L1D) + kron(L1D, I)

    A_u_lhs = identity(N*N) - 0.5 * dt * Du * L2D
    A_u_rhs = identity(N*N) + 0.5 * dt * Du * L2D
    A_v_lhs = identity(N*N) - 0.5 * dt * Dv * L2D
    A_v_rhs = identity(N*N) + 0.5 * dt * Dv * L2D

    u = u.flatten()
    v = v.flatten()

    solver_u = factorized(A_u_lhs)
    solver_v = factorized(A_v_lhs)
    
    start = time.time()

    for step in range(steps):
        Fu = f(u, v, p)
        Gv = g(u, v, p)

        rhs_u = A_u_rhs @ u + dt * Fu
        rhs_v = A_v_rhs @ v + dt * Gv

        #u = spsolve(A_u_lhs, rhs_u)
        #v = spsolve(A_v_lhs, rhs_v)
        u = solver_u(rhs_u)
        v = solver_v(rhs_v)
    
    print(f"Total time for {steps} steps: {time.time() - start:.2f} s")

    return u.reshape((N, N)), v.reshape((N, N))
