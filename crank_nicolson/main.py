import numpy as np
import matplotlib.pyplot as plt

from gray_scott import gray_scott_f, gray_scott_g
from crank_nicolson import crank_nicolson

# Initial condition setup
def initialize(N, perturb_type="random"):
    u = np.ones((N, N))
    v = np.zeros((N, N))
    r = N/4

    if perturb_type == "random":
        np.random.seed(0)
        v += 0.05 * np.random.rand(N, N)
    elif perturb_type == "none":
        u[N//2-r:N//2+r, N//2-r:N//2+r] = 0.50
        v[N//2-r:N//2+r, N//2-r:N//2+r] = 0.25
    return u, v

# Main execution
if __name__ == "__main__":
    N = 50
    dx = 1.0 / N
    dt = 1/2 # 1
    steps = 500 #1000
    Du, Dv = 0.16, 0.08

    p = [0.038, 0.061]  # alpha, beta
    # p = [0.0980, 0.0570]
    # p = [0.0620, 0.0610]
    # p = [0.0380, 0.0590]
    # p = [0.0220, 0.0490] 
    # p = [0.0060, 0.0350] 
    # p = [0.0100, 0.0510]

    # u0, v0 = initialize2(N)
    u0, v0 = initialize(N, perturb_type="random")
    u, v = crank_nicolson(u0, v0, Du, Dv, gray_scott_f, gray_scott_g, dt, dx, steps, p)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("u concentration")
    plt.imshow(u, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("v concentration")
    plt.imshow(v, cmap="inferno")
    plt.colorbar()
    plt.tight_layout()
    plt.show()