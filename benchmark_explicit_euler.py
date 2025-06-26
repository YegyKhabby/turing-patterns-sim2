
import numpy as np
import matplotlib.pyplot as plt

# Laplacian function
def laplacian(U, dx):
    L = np.zeros_like(U)
    L[1:-1, 1:-1] = (
        U[2:, 1:-1] + U[:-2, 1:-1] + U[1:-1, 2:] + U[1:-1, :-2] - 4 * U[1:-1, 1:-1]
    ) / dx**2
    return L

# Explicit Euler step
def explicit_euler(u, v, f, g, D_u, D_v, dx, dt):
    Lu = laplacian(u, dx)
    Lv = laplacian(v, dx)
    u_new = u + dt * (D_u * Lu + f(u, v))
    v_new = v + dt * (D_v * Lv + g(u, v))
    return u_new, v_new

# Gray-Scott parameters
alpha, beta = 0.04, 0.06
D_u, D_v = 0.1, 0.05
N = 100
dx = 1.0 / N
T = 1.0
dt_list = [1e-4, 5e-5, 2.5e-5]

# Reaction terms
def f(u, v): return -u * v**2 + alpha * (1 - u)
def g(u, v): return u * v**2 - (alpha + beta) * v

# Initial conditions
def initialize_uv(N):
    u = np.ones((N, N))
    v = np.zeros((N, N))
    r = 10
    u[N//2 - r:N//2 + r, N//2 - r:N//2 + r] = 0.5
    v[N//2 - r:N//2 + r, N//2 - r:N//2 + r] = 0.25
    return u, v

# Run the simulation
def run_simulation(dt):
    u, v = initialize_uv(N)
    steps = int(T / dt)
    for _ in range(steps):
        u, v = explicit_euler(u, v, f, g, D_u, D_v, dx, dt)
    return u, v

# Reference solution
ref_dt = dt_list[-1]
u_ref, v_ref = run_simulation(ref_dt)

# Error calculation
errors_u = []
errors_v = []

for dt in dt_list[:-1]:
    u_approx, v_approx = run_simulation(dt)
    errors_u.append(np.linalg.norm(u_approx - u_ref))
    errors_v.append(np.linalg.norm(v_approx - v_ref))

# Save plots
plt.imshow(u_ref, cmap='viridis')
plt.title(f'U field at T={T}s, dt={ref_dt}')
plt.colorbar()
plt.savefig("U_field_very_stable.png")
plt.close()

plt.imshow(v_ref, cmap='plasma')
plt.title(f'V field at T={T}s, dt={ref_dt}')
plt.colorbar()
plt.savefig("V_field_very_stable.png")
plt.close()

dt_vals = np.array(dt_list[:-1])
plt.loglog(dt_vals, errors_u, 'o-', label='Error in U')
plt.loglog(dt_vals, errors_v, 's-', label='Error in V')
plt.xlabel('dt')
plt.ylabel('L2 Error')
plt.title('Order of Accuracy (Explicit Euler - dt ≤ 1e-4)')
plt.legend()
plt.grid(True)
plt.savefig("Error_loglog_plot_very_stable.png")
plt.close()
