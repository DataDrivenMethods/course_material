"""Obizhaeva-Wang risk-neutral optimal schedules for several values of beta*T."""
import numpy as np
import matplotlib.pyplot as plt

cases = [0.5, 2, 5, 20, 100]   # beta*T
colors = plt.cm.viridis(np.linspace(0, 0.85, len(cases)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
tau = np.linspace(0, 1, 201)
for b, c in zip(cases, colors):
    d = b + 2
    # remaining inventory: jump at 0, linear on (0,T), jump at T
    t = np.concatenate(([0, 0], tau, [1]))
    x = np.concatenate(([1, (1 + b) / d], (1 + b * (1 - tau)) / d, [0]))
    ax1.plot(t, x, color=c, lw=2, label=rf"$\beta T = {b:g}$")
    # impact I_t / (lambda X_0): 0 before, 1/d on [0,T), 2/d after final block
    ti = [-0.05, 0, 0, 1, 1, 1.05]
    ii = [0, 0, 1 / d, 1 / d, 2 / d, 2 / d]
    ax2.plot(ti, ii, color=c, lw=2)

ax1.plot([0, 1], [1, 0], "k--", lw=1.2, label=r"TWAP ($\beta T \to \infty$)")
ax1.set(xlabel=r"$t/T$", ylabel=r"$X_t / X_0$", title="Remaining inventory", xlim=(-0.02, 1.02), ylim=(0, 1.02))
ax2.set(xlabel=r"$t/T$", ylabel=r"$I_t / (\lambda X_0)$", title="Price impact", xlim=(-0.05, 1.05), ylim=(0, 0.85))
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
ax1.legend(frameon=False)
fig.tight_layout()
fig.savefig("obizhaeva_wang_trajectories.png", dpi=200)
