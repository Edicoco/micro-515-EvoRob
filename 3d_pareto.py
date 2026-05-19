import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401 — registers the 3d projection


def pareto_mask(F):
    """Return boolean mask of non-dominated (Pareto-front) solutions (maximisation)."""
    n = len(F)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        # i is dominated if any j dominates it
        dominated[i] = np.any(
            np.all(F >= F[i], axis=1) & np.any(F > F[i], axis=1)
        )
    return ~dominated


data = np.load("NSGA/run_01/full_f.npy")
F = data[0]  # shape (512, 3)

mask = pareto_mask(F)
print(f"Population size : {len(F)}")
print(f"Pareto-front size: {mask.sum()}")

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

# All dominated solutions
ax.scatter(
    F[~mask, 0], F[~mask, 1], F[~mask, 2],
    c="steelblue", alpha=0.25, s=18, label=f"Dominated ({(~mask).sum()})",
)

# Pareto-front solutions
ax.scatter(
    F[mask, 0], F[mask, 1], F[mask, 2],
    c="crimson", alpha=0.9, s=45, edgecolors="black", linewidths=0.4,
    label=f"Pareto front ({mask.sum()})",
)

ax.set_xlabel("Flat fitness", labelpad=8)
ax.set_ylabel("Ice fitness", labelpad=8)
ax.set_zlabel("Hill fitness", labelpad=8)
ax.set_title("3-Objective Pareto Front\n(Flat · Ice · Hill terrains)", pad=14)
ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig("NSGA/run_01/pareto_3d.png", dpi=150)
plt.show()
print("Saved → .png")
