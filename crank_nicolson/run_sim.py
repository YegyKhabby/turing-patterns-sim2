import numpy as np
import matplotlib.pyplot as plt

from gray_scott import gray_scott_f, gray_scott_g
from crank_nicolson import crank_nicolson
from initialize import initialize

def run_simulation(N, dt, total_time, Du, Dv, params, pattern='random', size=10):
    dx = 1.0 / 143
    steps = int(total_time / dt)
    u0, v0 = initialize(N, pattern, size)
    return crank_nicolson(u0, v0, Du, Dv, gray_scott_f, gray_scott_g, dt, dx, steps, params)

def visualize(v0, v, total_time):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    #plt.title('u pattern')
    #plt.imshow(u, cmap='inferno')
    plt.title('initial')
    plt.imshow(v0, cmap='inferno')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title(f'v pattern, t = {total_time}')
    plt.imshow(v, cmap='inferno')
    plt.colorbar()
    plt.tight_layout()

    plt.show()

# Main execution
if __name__ == "__main__":
    Du, Dv = 0.16, 0.08
    p = [0.038, 0.061]  # alpha, beta
    u, v = run_simulation(N=64, dt=1/2, total_time=500, Du=Du, Dv=Dv, params=p)
    visualize(u, v)