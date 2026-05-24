"""
xy_plot.py — Three plots comparing robot trajectories before vs after training.

  1. XY trajectory map
  2. Bar chart — mean final-X distance with 95% CI
  3. Histogram — distribution of final-X distances (Gaussian-like)

Blue  = before training  (results/Before)
Red   = after  training  (results/After)

Each *.npy file is a (T, 2) array of (x, y) positions.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from evorob.utils.filesys import get_project_root

ROOT = get_project_root()

FOLDERS = {
    "before": {
        "path":  os.path.join(ROOT, "Before"),
        "color": "steelblue",
        "label": "Before training",
    },
    "after": {
        "path":  os.path.join(ROOT, "After"),
        "color": "tomato",
        "label": "After training",
    },
}

SAVE_DIR = os.path.join(ROOT, "results")


def load_trajectories(folder: str) -> list[np.ndarray]:
    """Return all trajectory arrays found in *folder* (root + recursive)."""
    patterns = [
        os.path.join(folder, "*.npy"),
        os.path.join(folder, "**", "*.npy"),
    ]
    seen, trajs = set(), []
    for pat in patterns:
        for f in sorted(glob.glob(pat, recursive=True)):
            if f in seen:
                continue
            seen.add(f)
            arr = np.load(f)
            if arr.ndim == 2 and arr.shape[1] == 2:
                trajs.append(arr)
            else:
                print(f"  Skipping {f}: shape {arr.shape}")
    return trajs


def final_x(trajs: list[np.ndarray]) -> np.ndarray:
    """Return array of final X positions, one per trajectory."""
    return np.array([t[-1, 0] for t in trajs])


# ---------------------------------------------------------------------------
# Plot 1 — XY trajectories
# ---------------------------------------------------------------------------

def plot_trajectories(data: dict, out_dir: str) -> None:
    _, ax = plt.subplots(figsize=(10, 6))
    legend_handles = []

    for cfg in data.values():
        trajs = cfg["trajs"]
        if not trajs:
            continue
        for i, traj in enumerate(trajs):
            ax.plot(
                traj[:, 0], traj[:, 1],
                color=cfg["color"], alpha=0.35, linewidth=0.8,
                label=cfg["label"] if i == 0 else "_nolegend_",
            )
        legend_handles.append(
            plt.Line2D([0], [0], color=cfg["color"], linewidth=2, label=cfg["label"])
        )

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Robot trajectories — before vs after training")
    ax.legend(handles=legend_handles)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(out_dir, "plot_trajectories.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2 — Bar chart with 95% CI on final X distance
# ---------------------------------------------------------------------------

def plot_bar_ci(data: dict, out_dir: str) -> None:
    _, ax = plt.subplots(figsize=(6, 5))

    labels, means, cis, colors = [], [], [], []
    for cfg in data.values():
        fx = cfg["final_x"]
        if len(fx) == 0:
            continue
        n   = len(fx)
        mu  = fx.mean()
        sem = fx.std(ddof=1) / np.sqrt(n)
        ci  = stats.t.ppf(0.975, df=n - 1) * sem  # 95% CI half-width
        labels.append(cfg["label"])
        means.append(mu)
        cis.append(ci)
        colors.append(cfg["color"])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=cis, color=colors, width=0.5,
                  capsize=8, error_kw={"elinewidth": 1.5, "ecolor": "black"})

    for bar, mu, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mu + ci + max(means) * 0.01,
                f"{mu:.2f} ± {ci:.2f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean final X distance (m)")
    ax.set_title("Mean distance reached — 95% confidence interval")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(out_dir, "plot_bar_ci.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3 — Histogram of final X distances
# ---------------------------------------------------------------------------

def plot_histogram(data: dict, out_dir: str) -> None:
    _, ax = plt.subplots(figsize=(8, 5))

    all_x = np.concatenate([cfg["final_x"] for cfg in data.values() if len(cfg["final_x"])])
    x_min, x_max = all_x.min(), all_x.max()
    bins = np.linspace(x_min, x_max, 25)

    for cfg in data.values():
        fx = cfg["final_x"]
        if len(fx) == 0:
            continue
        ax.hist(fx, bins=bins, color=cfg["color"], alpha=0.55,
                label=cfg["label"], edgecolor="white", linewidth=0.4)

        # Gaussian fit overlay
        mu, sigma = fx.mean(), fx.std(ddof=1)
        x_fit = np.linspace(x_min, x_max, 300)
        bin_width = bins[1] - bins[0]
        scale = len(fx) * bin_width  # scale PDF to counts
        y_fit = stats.norm.pdf(x_fit, mu, sigma) * scale
        ax.plot(x_fit, y_fit, color=cfg["color"], linewidth=2, linestyle="--")

    ax.set_xlabel("Final X distance (m)")
    ax.set_ylabel("Number of robots")
    ax.set_title("Distribution of final X distance — before vs after training")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(out_dir, "plot_histogram.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load all trajectories once
    data = {}
    for key, cfg in FOLDERS.items():
        trajs = load_trajectories(cfg["path"])
        fx    = final_x(trajs)
        print(f"  {cfg['label']}: {len(trajs)} trajectories"
              + (f"  mean_x={fx.mean():.2f}" if len(fx) else "  (empty)"))
        data[key] = {**cfg, "trajs": trajs, "final_x": fx}

    plot_trajectories(data, SAVE_DIR)
    plot_bar_ci(data, SAVE_DIR)
    plot_histogram(data, SAVE_DIR)


if __name__ == "__main__":
    main()
