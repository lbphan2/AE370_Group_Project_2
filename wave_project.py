import numpy as np
import matplotlib.pyplot as plt
import niceplots

# Plot styling
plt.style.use(niceplots.get_style("james-light"))
colors = niceplots.get_colors()


# 1. SOLVER IMPLEMENTATION
def analytical_solution(x, t, L, c):
    return np.sin(np.pi * x / L) * np.cos(np.pi * c * t / L)

def solve_wave_equation(Nx, CFL, T_end, L=1.0, c=1.0):
    dx = L / Nx
    dt = CFL * dx / c
    Nt = int(np.ceil(T_end / dt))
    dt = T_end / Nt
    C_actual = c * dt / dx

    x = np.linspace(0, L, Nx + 1)

    # Initial condition
    u_curr = np.sin(np.pi * x / L)
    u_prev = u_curr.copy()
    u_next = np.zeros_like(u_curr)

    sigma_sq = C_actual**2

    # First time step (Taylor expansion, u_t = 0)
    for i in range(1, Nx):
        u_next[i] = u_curr[i] + 0.5 * sigma_sq * (
            u_curr[i+1] - 2*u_curr[i] + u_curr[i-1]
        )
    u_next[0] = 0.0
    u_next[-1] = 0.0

    u_prev[:] = u_curr
    u_curr[:] = u_next

    # Time marching
    for _ in range(1, Nt):
        u_next[1:-1] = (
            2*u_curr[1:-1] - u_prev[1:-1]
            + sigma_sq * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2])
        )
        u_next[0] = 0.0
        u_next[-1] = 0.0

        u_prev[:] = u_curr
        u_curr[:] = u_next

    u_exact = analytical_solution(x, T_end, L, c)
    return x, u_curr, u_exact, dt


# 2. CONVERGENCE STUDY
def run_convergence_study():
    L = 1.0
    c = 1.0
    T_end = 1.0
    CFL = 0.5

    grid_resolutions = [20, 40, 80, 160, 320]
    errors = []
    dx_values = []

    print("-" * 50)
    print("CONVERGENCE STUDY RESULTS")
    print(f"{'Nx':<10} | {'dx':<10} | {'L2 Error':<15} | {'Order'}")
    print("-" * 50)

    last_order_str = "N/A"
    for i, Nx in enumerate(grid_resolutions):
        x, u_num, u_exact, _ = solve_wave_equation(Nx, CFL, T_end, L, c)
        error = np.sqrt(np.mean((u_num - u_exact)**2))
        dx = L / Nx

        errors.append(error)
        dx_values.append(dx)

        order_str = "N/A"
        if i > 0:
            order = np.log(errors[i-1] / errors[i]) / np.log(dx_values[i-1] / dx_values[i])
            order_str = f"{order:.2f}"
            last_order_str = order_str

        print(f"{Nx:<10} | {dx:<10.4f} | {error:<15.2e} | {order_str}")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.loglog(dx_values, errors, marker="o", linewidth=2, label="Simulation Error")
    ref_x = np.array(dx_values)
    ref_y = errors[-1] * (ref_x / ref_x[-1])**2
    ax.loglog(ref_x, ref_y, "--", linewidth=2, label=r"Reference $O(\Delta x^2)$")

    ax.text(dx_values[1], errors[1]*1.5,
            rf"Observed Order $\approx {last_order_str}$", fontsize=12)

    ax.set_xlabel(r"Grid Spacing $\Delta x$")
    ax.set_ylabel(r"$L_2$ Error Norm")
    ax.set_title(f"Spatial Convergence Study (CFL = {CFL})", pad=8)
    ax.grid(True, which="both", alpha=0.35)
    ax.invert_xaxis()

    # Legend inside, normal, unobtrusive
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    plt.show()


# 3. STABILITY STUDY (SEPARATE FIGURES)
def run_stability_evaluation():
    L = 1.0
    c = 1.0
    T_end = 2.0
    Nx = 50

    cfl_cases = [0.5, 0.9, 1.05, 1.2]

    for cfl in cfl_cases:
        x, u_num, u_exact, _ = solve_wave_equation(Nx, cfl, T_end, L, c)

        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.plot(x, u_exact, "--", linewidth=2, label="Exact")
        if cfl <= 1.0:
            ax.plot(x, u_num, marker="o", linewidth=2,
                    markersize=4, label="Numerical (Stable)")
            ax.set_ylim(-1.5, 1.5)
            ax.set_title(f"Stable Case (CFL = {cfl})", pad=8)
        else:
            ax.plot(x, u_num, marker="x", linewidth=2,
                    markersize=4, label="Numerical (Unstable)")
            max_val = np.max(np.abs(u_num))
            ax.text(
                0.02, 0.95, f"Max |u| = {max_val:.1e}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=11,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.9)
            )
            ax.set_title(f"Unstable Case (CFL = {cfl})", pad=8)

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u(x,t)$")
        ax.grid(True, alpha=0.35)

        # Legend inside axes; choose corner that usually won't cover the peak
        ax.legend(loc="lower left", frameon=False)

        # Remove the extra suptitle (this was creating the big empty gap)
        fig.tight_layout()
        plt.show()


# MAIN EXECUTION
if __name__ == "__main__":
    run_convergence_study()
    run_stability_evaluation()
