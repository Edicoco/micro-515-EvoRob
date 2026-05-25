"""
specialist_statistic.py — Cross-evaluation of 3 specialists on 3 environments.

Evaluates flat_specialist, ice_specialist, and hill_specialist — each genome
unchanged — on FlatEnv-v0, IceEnv-v0, and HillEnv-v0.
Runs N_RUNS independent episodes per (specialist × environment) cell and tests
whether each specialist performs significantly better on its home environment
than the other two (one-sided Mann-Whitney U test, α=0.05).

Outputs (saved to results/specialist_stats/):
  • heatmap_mean.png       — 3×3 mean-fitness heatmap
  • boxplots.png           — box plots per environment
  • barplot_ci.png         — bar chart with 95% CI per environment
  • significance_table.png — p-value annotation table
  • results.npz            — raw scores for later analysis

Usage:
    python specialist_statistic.py \
        --flat_dir results/flat_specialist_cmaes_long/final \
        --ice_dir  results/ice_specialist_cmaes_short/final \
        --hill_dir results/Hill_specialist_cmaes_long/final
"""

import argparse
import os
from os.path import join

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import evorob.world  # noqa: F401 — registers all envs
from evorob.utils.filesys import get_project_root
from final_project_train import FinalWorld

ROOT_DIR  = get_project_root()
N_WEIGHTS = 560
N_RUNS    = 40    # independent episodes per cell
N_STEPS   = 1000   # steps per episode (shorter for speed)
ALPHA     = 0.05

SPECIALIST_KEYS = ["flat", "ice", "hill"]
ENV_KEYS        = ["flat", "ice", "hill"]

SPECIALIST_COLORS = {
    "flat": "#4C72B0",
    "ice":  "#55A868",
    "hill": "#C44E52",
}
ENV_LABELS = {
    "flat": "Flat",
    "ice":  "Ice",
    "hill": "Hill",
}
SPECIALIST_LABELS = {
    "flat": "Flat specialist",
    "ice":  "Ice specialist",
    "hill": "Hill specialist",
}

OUT_DIR = join(ROOT_DIR, "results", "specialist_stats")


# ---------------------------------------------------------------------------
# Body params (standard ant morphology, same for all specialists)
# ---------------------------------------------------------------------------

def _standard_body() -> np.ndarray:
    return np.array([0.05, 0.9, 0.05, 0.9])


# ---------------------------------------------------------------------------
# Genome loading
# ---------------------------------------------------------------------------

def load_genome(specialist_dir: str) -> np.ndarray:
    path = join(specialist_dir, "x_best.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"x_best.npy not found in {specialist_dir}")
    genome = np.load(path)[:N_WEIGHTS]
    print(f"  Loaded genome from {path}  shape={genome.shape}")
    return genome


# ---------------------------------------------------------------------------
# Single-episode evaluation
# ---------------------------------------------------------------------------

_ENV_MAP = {
    "flat": "FlatEnv-v0",
    "ice":  "IceEnv-v0",
    "hill": "HillEnv-v0",
}


def evaluate_episode(world: FinalWorld, env_key: str) -> tuple[float, float]:
    """Run one episode on *env_key*; return (total_reward, distance_traveled_x)."""
    import gymnasium as gym
    env_id = _ENV_MAP[env_key]
    world_file = {
        "flat": world.flat_world_file,
        "ice":  world.ice_world_file,
        "hill": world.hill_world_file,
    }[env_key]
    env = gym.make(env_id, robot_path=world_file, max_episode_steps=N_STEPS)
    world.controller.reset_controller(batch_size=1)
    obs, reset_info = env.reset()
    x_start = float(reset_info.get("x_position", 0.0))
    total_reward = 0.0
    x_final = x_start
    done = False
    while not done:
        action = world.controller.get_action(obs)
        if action.ndim > 1:
            action = action.squeeze(0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        x_final = float(info.get("x_position", x_final))
        done = terminated or truncated
    env.close()
    return total_reward, x_final - x_start


# ---------------------------------------------------------------------------
# Full evaluation grid
# ---------------------------------------------------------------------------

def run_all(genomes: dict[str, np.ndarray]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Returns scores[specialist_key][env_key] = {"reward": np.ndarray, "distance": np.ndarray},
    each of shape (N_RUNS,).
    """
    scores: dict[str, dict[str, dict[str, list]]] = {
        s: {e: {"reward": [], "distance": []} for e in ENV_KEYS} for s in SPECIALIST_KEYS
    }

    for sp_key in SPECIALIST_KEYS:
        print(f"\n── Evaluating {SPECIALIST_LABELS[sp_key]} ──")
        world = FinalWorld()
        genome_full = np.concatenate([genomes[sp_key], _standard_body()])
        world.update_robot_xml(genome_full)

        for env_key in ENV_KEYS:
            print(f"   env={ENV_LABELS[env_key]}  ", end="", flush=True)
            for _ in range(N_RUNS):
                reward, distance = evaluate_episode(world, env_key)
                scores[sp_key][env_key]["reward"].append(reward)
                scores[sp_key][env_key]["distance"].append(distance)
                print(".", end="", flush=True)
            mean_r = np.mean(scores[sp_key][env_key]["reward"])
            mean_d = np.mean(scores[sp_key][env_key]["distance"])
            print(f"  reward={mean_r:.1f}  dist={mean_d:.1f}")

        world.temp_dir.cleanup()

    return {
        sp: {
            env: {
                "reward":   np.array(v["reward"]),
                "distance": np.array(v["distance"]),
            }
            for env, v in envs.items()
        }
        for sp, envs in scores.items()
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_stats(scores: dict) -> dict:
    """
    For each environment, test home_specialist > each_other with two
    one-sided Mann-Whitney U tests: one on reward, one on distance.

    Returns results[env_key] = {
        "home": specialist_key,
        "vs_other": {
            "reward":   {"stat", "p", "sig"},
            "distance": {"stat", "p", "sig"},
        }, ...
    }
    """
    results = {}
    for env_key in ENV_KEYS:
        home   = env_key
        others = [s for s in SPECIALIST_KEYS if s != home]
        entry  = {"home": home}
        for other in others:
            tests = {}
            for metric in ("reward", "distance"):
                stat, p = stats.mannwhitneyu(
                    scores[home][env_key][metric],
                    scores[other][env_key][metric],
                    alternative="greater",
                )
                tests[metric] = {"stat": stat, "p": p, "sig": p < ALPHA}
            entry[f"vs_{other}"] = tests
        results[env_key] = entry
    return results


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Plot 1 — Heatmap (reward | distance), side by side
# ---------------------------------------------------------------------------

def plot_heatmap(scores: dict, stat_results: dict, out_dir: str) -> None:
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, cbar_label, title in [
        (axes[0], "reward",   "Mean reward",   "Mean reward — specialists × environments"),
        (axes[1], "distance", "Mean distance", "Mean distance — specialists × environments"),
    ]:
        matrix = np.array([
            [scores[sp][env][metric].mean() for env in ENV_KEYS]
            for sp in SPECIALIST_KEYS
        ])
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xticks(range(len(ENV_KEYS)))
        ax.set_yticks(range(len(SPECIALIST_KEYS)))
        ax.set_xticklabels([ENV_LABELS[e] for e in ENV_KEYS], fontsize=11)
        ax.set_yticklabels([SPECIALIST_LABELS[s] for s in SPECIALIST_KEYS], fontsize=11)
        ax.set_xlabel("Environment", fontsize=12)
        ax.set_ylabel("Specialist", fontsize=12)
        ax.set_title(title, fontsize=12)

        vmin, vmax = matrix.min(), matrix.max()
        for i, sp in enumerate(SPECIALIST_KEYS):
            for j, env in enumerate(ENV_KEYS):
                val = matrix[i, j]
                star = ""
                if sp == env:
                    entry = stat_results[env]
                    stars_list = [
                        sig_stars(entry[f"vs_{o}"][metric]["p"])
                        for o in SPECIALIST_KEYS if o != sp
                    ]
                    star = " " + "/".join(s for s in stars_list if s != "ns") or ""
                rel = (val - vmin) / (vmax - vmin + 1e-9)
                color = "black" if 0.2 < rel < 0.8 else "white"
                ax.text(j, i, f"{val:.1f}{star}", ha="center", va="center",
                        fontsize=10, color=color)

        for k in range(len(SPECIALIST_KEYS)):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                       fill=False, edgecolor="gold", linewidth=3))

    plt.tight_layout()
    path = join(out_dir, "heatmap_mean.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2 — Box plots (2 rows × 3 env cols: reward / distance)
# ---------------------------------------------------------------------------

def _draw_box_env(ax, scores, stat_results, env_key, metric, ylabel):
    data   = [scores[sp][env_key][metric] for sp in SPECIALIST_KEYS]
    colors = [SPECIALIST_COLORS[sp] for sp in SPECIALIST_KEYS]
    bp     = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(SPECIALIST_KEYS) + 1))
    ax.set_xticklabels([SPECIALIST_LABELS[s] for s in SPECIALIST_KEYS],
                       fontsize=8, rotation=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    home_idx = SPECIALIST_KEYS.index(env_key)
    entry    = stat_results[env_key]
    y_max    = max(d.max() for d in data)
    y_range  = y_max - min(d.min() for d in data)
    offset   = y_range * 0.06
    for other_key in [s for s in SPECIALIST_KEYS if s != env_key]:
        other_idx = SPECIALIST_KEYS.index(other_key)
        p_val = entry[f"vs_{other_key}"][metric]["p"]
        star  = sig_stars(p_val)
        x1, x2 = sorted([home_idx + 1, other_idx + 1])
        y_b = y_max + offset
        ax.annotate("", xy=(x2, y_b), xytext=(x1, y_b),
                    arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
        ax.text((x1 + x2) / 2, y_b + offset * 0.3, star,
                ha="center", va="bottom", fontsize=10,
                color="green" if star != "ns" else "grey")
        offset += y_range * 0.10


def plot_boxplots(scores: dict, stat_results: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharey=False)

    for col, env_key in enumerate(ENV_KEYS):
        axes[0, col].set_title(f"{ENV_LABELS[env_key]} environment", fontsize=11)
        _draw_box_env(axes[0, col], scores, stat_results, env_key,
                      "reward",   "Reward" if col == 0 else "")
        _draw_box_env(axes[1, col], scores, stat_results, env_key,
                      "distance", "Distance" if col == 0 else "")

    legend_patches = [
        mpatches.Patch(color=SPECIALIST_COLORS[s], label=SPECIALIST_LABELS[s], alpha=0.75)
        for s in SPECIALIST_KEYS
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Reward (top) and Distance (bottom) per environment\n"
                 "(brackets: home specialist > other, one-sided MWU)",
                 fontsize=12, y=1.04)

    plt.tight_layout()
    path = join(out_dir, "boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3 — Bar chart with 95% CI (reward | distance), side by side
# ---------------------------------------------------------------------------

def plot_bar_ci(scores: dict, out_dir: str) -> None:
    n_sp    = len(SPECIALIST_KEYS)
    x       = np.arange(len(ENV_KEYS))
    width   = 0.22
    offsets = np.linspace(-(n_sp - 1) * width / 2, (n_sp - 1) * width / 2, n_sp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, title in [
        (axes[0], "reward",   "Mean reward ± 95% CI",   "Reward — specialists × environments"),
        (axes[1], "distance", "Mean distance ± 95% CI", "Distance — specialists × environments"),
    ]:
        for sp_key, off in zip(SPECIALIST_KEYS, offsets):
            means, cis = [], []
            for env_key in ENV_KEYS:
                arr = scores[sp_key][env_key][metric]
                n   = len(arr)
                mu  = arr.mean()
                ci  = stats.t.ppf(0.975, df=n - 1) * arr.std(ddof=1) / np.sqrt(n)
                means.append(mu)
                cis.append(ci)
            ax.bar(x + off, means, width=width,
                   color=SPECIALIST_COLORS[sp_key], alpha=0.8,
                   label=SPECIALIST_LABELS[sp_key],
                   yerr=cis, capsize=5,
                   error_kw={"elinewidth": 1.2, "ecolor": "black"})
        ax.set_xticks(x)
        ax.set_xticklabels([ENV_LABELS[e] for e in ENV_KEYS], fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = join(out_dir, "barplot_ci.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 4 — Significance table (reward + distance columns)
# ---------------------------------------------------------------------------

def plot_significance_table(stat_results: dict, out_dir: str) -> None:
    rows, row_labels = [], []
    # column headers: for each other specialist, two cols (reward / distance)
    first_env  = ENV_KEYS[0]
    others_ref = [s for s in SPECIALIST_KEYS if s != first_env]
    col_labels  = [f"{lbl}\n{metric}"
                   for lbl in [f"vs {SPECIALIST_LABELS[o]}" for o in others_ref]
                   for metric in ("Reward", "Distance")]

    for env_key in ENV_KEYS:
        entry  = stat_results[env_key]
        home   = entry["home"]
        others = [s for s in SPECIALIST_KEYS if s != home]
        row_data = []
        for other in others:
            for metric in ("reward", "distance"):
                info = entry[f"vs_{other}"][metric]
                row_data.append(f"p={info['p']:.4f}  {sig_stars(info['p'])}")
        rows.append(row_data)
        row_labels.append(f"{ENV_LABELS[env_key]}\n(home: {SPECIALIST_LABELS[home]})")

    fig, ax = plt.subplots(figsize=(13, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.3, 2.2)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#DDDDDD")
        else:
            text = cell.get_text().get_text()
            if "***" in text or "**" in text or "*" in text:
                cell.set_facecolor("#C8E6C9")
            else:
                cell.set_facecolor("#FFCDD2")

    ax.set_title("One-sided MWU: home specialist > other  (reward | distance)\n"
                 "(green = significant at α=0.05, red = not significant)",
                 fontsize=11, pad=20)

    plt.tight_layout()
    path = join(out_dir, "significance_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(scores: dict, stat_results: dict) -> None:
    for metric in ("reward", "distance"):
        label = "REWARD" if metric == "reward" else "DISTANCE"
        print("\n" + "=" * 72)
        print(f"{'CROSS-EVALUATION SUMMARY — ' + label:^72}")
        print("=" * 72)
        header = f"{'':22}" + "".join(f"{ENV_LABELS[e]:>14}" for e in ENV_KEYS)
        print(header)
        print("-" * 72)
        for sp in SPECIALIST_KEYS:
            row = f"{SPECIALIST_LABELS[sp]:22}"
            for env in ENV_KEYS:
                arr  = scores[sp][env][metric]
                mark = " ✓" if sp == env else "  "
                row += f"{arr.mean():>12.1f}{mark}"
            print(row)

        print(f"\nStatistical tests on {label} (one-sided MWU, home > other):")
        print("-" * 72)
        for env_key in ENV_KEYS:
            entry  = stat_results[env_key]
            home   = entry["home"]
            others = [s for s in SPECIALIST_KEYS if s != home]
            for other in others:
                info   = entry[f"vs_{other}"][metric]
                result = "SIGNIFICANT" if info["sig"] else "not significant"
                print(f"  {ENV_LABELS[env_key]:5s}  {SPECIALIST_LABELS[home]:20s} > "
                      f"{SPECIALIST_LABELS[other]:20s}  "
                      f"p={info['p']:.4f}  {sig_stars(info['p'])}  [{result}]")
        print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(flat_dir: str, ice_dir: str, hill_dir: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nLoading genomes...")
    genomes = {
        "flat": load_genome(flat_dir),
        "ice":  load_genome(ice_dir),
        "hill": load_genome(hill_dir),
    }

    print(f"\nEvaluating {N_RUNS} runs × 3 specialists × 3 environments "
          f"({N_RUNS * 9} total episodes, {N_STEPS} steps each)...")
    scores = run_all(genomes)

    np.savez(join(OUT_DIR, "results.npz"), **{
        f"{sp}_{env}_{metric}": scores[sp][env][metric]
        for sp in SPECIALIST_KEYS for env in ENV_KEYS for metric in ("reward", "distance")
    })
    print(f"  Raw scores saved: {join(OUT_DIR, 'results.npz')}")

    stat_results = run_stats(scores)
    print_summary(scores, stat_results)

    print("\nGenerating plots...")
    plot_heatmap(scores, stat_results, OUT_DIR)
    plot_boxplots(scores, stat_results, OUT_DIR)
    plot_bar_ci(scores, OUT_DIR)
    plot_significance_table(stat_results, OUT_DIR)

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flat_dir", type=str,
        # default=join(ROOT_DIR, "results/randomized_generalist/IT_001/990"),
        default=join(ROOT_DIR, "results/randomized_generalist/flat_stat/450"),    
        help="Directory containing flat specialist x_best.npy",
    )
    parser.add_argument(
        "--ice_dir", type=str,
        default=join(ROOT_DIR, "results/randomized_generalist/IT_002/final"),
        help="Directory containing ice specialist x_best.npy",
    )
    parser.add_argument(
        "--hill_dir", type=str,
        default=join(ROOT_DIR, "results/randomized_generalist/hill_stat/460"),
        help="Directory containing hill specialist x_best.npy",
    )
    parser.add_argument("--n_runs",  type=int, default=N_RUNS,
                        help="Independent episodes per (specialist, env) cell")
    parser.add_argument("--n_steps", type=int, default=N_STEPS,
                        help="Steps per episode")
    args = parser.parse_args()

    N_RUNS  = args.n_runs
    N_STEPS = args.n_steps

    main(flat_dir=args.flat_dir, ice_dir=args.ice_dir, hill_dir=args.hill_dir)
