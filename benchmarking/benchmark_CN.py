import numpy as np
import matplotlib.pyplot as plt
import time 

from solvers.crank_nicolson import crank_nicolson
from models.gray_scott import gray_scott_f, gray_scott_g
from models.initialize import initialize

def benchmark_crank_nicolson(Du, Dv, params, dt=1.0, total_time=100.0,
                             Ns=[32, 64, 96, 128], step_counts=[50, 100, 200, 400]):
    # --- Benchmark vs Grid Size ---
    runtime_vs_N = []
    for N in Ns:
        dx = 1.0 / N
        steps = int(total_time / dt)
        u0, v0 = initialize(N, pattern='gradient')
        print(f"Benchmarking CN: grid size N = {N}")
        start = time.time()
        crank_nicolson(u0, v0, Du, Dv, gray_scott_f, gray_scott_g, dt, dx, steps, params)
        runtime_vs_N.append(time.time() - start)

    # --- Benchmark vs Time Steps ---
    N_fixed = 64
    dx = 1.0 / N_fixed
    runtime_vs_steps = []
    u0, v0 = initialize(N_fixed, pattern='gradient')
    for steps in step_counts:
        print(f"Benchmarking CN: time steps = {steps}")
        start = time.time()
        crank_nicolson(u0, v0, Du, Dv, gray_scott_f, gray_scott_g, dt, dx, steps, params)
        runtime_vs_steps.append(time.time() - start)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))

    # Bottom x-axis: Grid size
    ax.set_xlabel("Grid Size N", color='tab:blue')
    ax.set_ylabel("Runtime (seconds)")
    ax.plot(Ns, runtime_vs_N, 'o-', color='tab:blue', label='Runtime vs Grid Size (N)')
    ax.tick_params(axis='x', colors='tab:blue')

    # Add scaling line O(N^2)
    N_ref = np.array(Ns)
    scale_N = runtime_vs_N[0] / N_ref[0]**2
    ax.plot(N_ref, scale_N * N_ref**2, 'k--', color='tab:blue', label=r'$\mathcal{O}(N^2)$')

    # Top x-axis: Steps
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(np.linspace(Ns[0], Ns[-1], len(step_counts)))
    ax_top.set_xticklabels(step_counts)
    ax_top.set_xlabel("Number of Time Steps", color='tab:red')
    ax_top.tick_params(axis='x', colors='tab:red')

    # Runtime vs steps (on same y-axis)
    x_steps_scaled = np.linspace(Ns[0], Ns[-1], len(step_counts))
    ax.plot(x_steps_scaled, runtime_vs_steps, 's-', color='tab:red', label='Runtime vs Steps')

    # Add O(steps) scaling line
    steps_ref = np.array(step_counts)
    scale_steps = runtime_vs_steps[0] / steps_ref[0]
    ax.plot(x_steps_scaled, scale_steps * steps_ref, 'r:', label=r'$\mathcal{O}(steps)$')

    # Legend
    ax.legend(loc='upper left')
    fig.suptitle("Crank–Nicolson Runtime Benchmark (Grid Size & Steps)")
    fig.tight_layout()
    plt.savefig("benchmark_crank_nicolson.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    Du, Dv = 0.16, 0.08
    params = [0.038, 0.061]  # alpha, beta
    benchmark_crank_nicolson(Du, Dv, params, dt=1.0, total_time=100.0,
                              Ns=[32, 64, 96, 128], step_counts=[50, 100, 200, 400])
    