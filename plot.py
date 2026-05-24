"""Plot fitness evolution over training generations for a results directory."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_fitness_history(run_dir: str):
    """Load per-checkpoint fitness arrays, return (gens, best_per_gen, mean_per_gen)."""
    gen_dirs = sorted(
        [d for d in os.listdir(run_dir) if d.isdigit()],
        key=int,
    )
    gens, best_vals, mean_vals = [], [], []
    for gd in gen_dirs:
        f_path = os.path.join(run_dir, gd, "f.npy")
        if not os.path.isfile(f_path):
            continue
        f = np.load(f_path)
        gens.append(int(gd))
        best_vals.append(float(np.max(f)))
        mean_vals.append(float(np.mean(f)))
    return np.array(gens), np.array(best_vals), np.array(mean_vals)


def plot_fitness(run_dir: str, save_path: str | None = None):
    label = os.path.basename(run_dir.rstrip("/"))
    gens, best, mean = load_fitness_history(run_dir)

    # Running best-so-far (monotonically non-decreasing)
    best_so_far = np.maximum.accumulate(best)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(gens, mean, best, alpha=0.15, color="steelblue", label="_nolegend_")
    ax.plot(gens, mean,         color="steelblue", lw=1.4, alpha=0.8, label="Generation mean")
    ax.plot(gens, best,         color="steelblue", lw=1.4, ls="--",   label="Generation best")
    ax.plot(gens, best_so_far,  color="crimson",   lw=2.2,             label="Best so far")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_title(f"Fitness evolution — {label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    from evorob.utils.filesys import get_project_root

    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?",
                        default=os.path.join(get_project_root(), "results/randomized_generalist/IT_01"))
    parser.add_argument("--save", type=str, default=None, help="Save figure to this path instead of showing")
    args = parser.parse_args()

    plot_fitness(args.run_dir, save_path=args.save)
