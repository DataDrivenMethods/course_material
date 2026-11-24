"""Almgren-Chriss optimal trajectories for several values of kappa*T."""
import numpy as np
import matplotlib.pyplot as plt

tau = np.linspace(0, 1, 401)  # t/T
cases = [0, 1, 2, 5, 10]       # kappa*T
colors = plt.cm.viridis(np.linspace(0, 0.85, len(cases)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
for k, c in zip(cases, colors):
    if k == 0:
        x = 1 - tau
        v = np.ones_like(tau)
        lab = r"$\kappa T = 0$ (TWAP)"
    else:
        x = np.sinh(k * (1 - tau)) / np.sinh(k)
        v = k * np.cosh(k * (1 - tau)) / np.sinh(k)
        lab = rf"$\kappa T = {k}$"
    ax1.plot(tau, x, color=c, lw=2, label=lab)
    ax2.plot(tau, v, color=c, lw=2, label=lab)

ax1.set(xlabel=r"$t/T$", ylabel=r"$X_t / X_0$", title="Remaining inventory", xlim=(0, 1), ylim=(0, 1.02))
ax2.set(xlabel=r"$t/T$", ylabel=r"$v_t T / X_0$", title="Trading rate", xlim=(0, 1), yscale="log")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
ax1.legend(frameon=False)
fig.tight_layout()
fig.savefig("almgren_chriss_trajectories.png", dpi=200)
