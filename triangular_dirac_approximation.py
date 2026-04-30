# triangular_dirac_approximation.py
import numpy as np
import matplotlib.pyplot as plt


def triangular_target(x, n):
    """Target triangular pulse with unit area."""
    return np.maximum(0.0, n - n**2 * np.abs(x))

## Hello


def fourier_partial_sum(x, L, n, m):
    """Fourier cosine partial sum for the periodic triangular pulse."""
    result = np.full_like(x, 1.0 / (2.0 * L), dtype=float)

    for k in range(1, m + 1):
        coefficient = (2.0 * L * n**2) / (k**2 * np.pi**2)
        coefficient *= 1.0 - np.cos(k * np.pi / (L * n))
        result += coefficient * np.cos(k * np.pi * x / L)

    return result


def plot_fourier_convergence(L=2, n=10, m_values=range(1, 11), num_points=4000):
    x = np.linspace(-L, L, num_points)
    target = triangular_target(x, n)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, m in zip(axes, m_values):
        partial = fourier_partial_sum(x, L=L, n=n, m=m)
        ax.plot(x, target, color="black", linestyle="--", linewidth=2, label="target")
        ax.plot(x, partial, color="tab:blue", linewidth=2, label=fr"$m={m}$")
        ax.set_title(f"m = {m}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-L, L)
        ax.set_ylim(-1.0, max(n + 0.5, 1.5))

    for ax in axes[5:]:
        ax.set_xlabel("x")
    for ax in axes[::5]:
        ax.set_ylabel(r"$f_{n,m}(x)$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(
        "Fourier Partial Sums Approaching the Triangular Dirac Approximation\n"
        f"(L = {L}, n = {n})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 0.96, 0.92))


def main():
    plot_fourier_convergence(L=2, n=10, m_values=(1, 2, 3, 4, 5, 10, 20, 50, 100, 200))
    plot_fourier_convergence(L=2, n=20, m_values=(1, 2, 3, 4, 5, 10, 20, 50, 100, 200))
    plot_fourier_convergence(L=2, n=50, m_values=(1, 2, 3, 4, 5, 10, 20, 50, 100, 200))
    plt.show()


if __name__ == "__main__":
    main()
