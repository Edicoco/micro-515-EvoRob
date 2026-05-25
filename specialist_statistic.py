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
N_RUNS    = 20    # independent episodes per cell
N_STEPS   = 500   # steps per episode (shorter for speed)
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

def evaluate_one(world: FinalWorld, env_key: str) -> float:
    """Run one episode on *env_key* and return total reward."""
    env_map = {
        "flat": ("FlatEnv-v0", world.flat_world_file),
        "ice":  ("IceEnv-v0",  world.ice_world_file),
        "hill": ("HillEnv-v0", world.hill_world_file),
    }
    env_id, world_file = env_map[env_key]
    return world._run_env(env_id, world_file, n_repeats=1, n_steps=N_STEPS)


# ---------------------------------------------------------------------------
# Full evaluation grid
# ---------------------------------------------------------------------------

def run_all(genomes: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """
    Returns scores[specialist_key][env_key] = np.ndarray of shape (N_RUNS,).
    """
    scores: dict[str, dict[str, list]] = {
        s: {e: [] for e in ENV_KEYS} for s in SPECIALIST_KEYS
    }

    for sp_key in SPECIALIST_KEYS:
        print(f"\n── Evaluating {SPECIALIST_LABELS[sp_key]} ──")
        world = FinalWorld()
        genome_full = np.concatenate([genomes[sp_key], _standard_body()])
        world.update_robot_xml(genome_full)

        for env_key in ENV_KEYS:
            print(f"   env={ENV_LABELS[env_key]}  ", end="", flush=True)
            for run in range(N_RUNS):
                score = evaluate_one(world, env_key)
                scores[sp_key][env_key].append(score)
                print(".", end="", flush=True)
            mean = np.mean(scores[sp_key][env_key])
            print(f"  mean={mean:.1f}")

        world.temp_dir.cleanup()

    return {sp: {env: np.array(v) for env, v in envs.items()}
            for sp, envs in scores.items()}


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_stats(scores: dict) -> dict:
    """
    For each environment, test home_specialist > each_other with
    one-sided Mann-Whitney U (alternative='greater').

    Returns results[env_key] = {
        "home": specialist_key,
        "vs_specialist1": {"stat": ..., "p": ..., "sig": bool},
        "vs_specialist2": {"stat": ..., "p": ..., "sig": bool},
    }
    """
    results = {}
    for env_key in ENV_KEYS:
        home = env_key  # home specialist == same key as env
        others = [s for s in SPECIALIST_KEYS if s != home]
        home_scores = scores[home][env_key]
        entry = {"home": home}
        for other in others:
            other_scores = scores[other][env_key]
            stat, p = stats.mannwhitneyu(
                home_scores, other_scores, alternative="greater"
            )
            entry[f"vs_{other}"] = {"stat": stat, "p": p, "sig": p < ALPHA}
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
# Plot 1 — Heatmap of mean fitness
# ---------------------------------------------------------------------------

def plot_heatmap(scores: dict, stat_results: dict, out_dir: str) -> None:
    matrix = np.array([
        [scores[sp][env].mean() for env in ENV_KEYS]
        for sp in SPECIALIST_KEYS
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean fitness")

    ax.set_xticks(range(len(ENV_KEYS)))
    ax.set_yticks(range(len(SPECIALIST_KEYS)))
    ax.set_xticklabels([ENV_LABELS[e] for e in ENV_KEYS], fontsize=11)
    ax.set_yticklabels([SPECIALIST_LABELS[s] for s in SPECIALIST_KEYS], fontsize=11)
    ax.set_xlabel("Environment", fontsize=12)
    ax.set_ylabel("Specialist", fontsize=12)
    ax.set_title("Mean fitness — specialists × environments", fontsize=13)

    for i, sp in enumerate(SPECIALIST_KEYS):
        for j, env in enumerate(ENV_KEYS):
            val = matrix[i, j]
            # Star annotation on home cell
            star = ""
            if sp == env:
                entry = stat_results[env]
                stars_list = [
                    sig_stars(entry[f"vs_{o}"]["p"])
                    for o in SPECIALIST_KEYS if o != sp
                ]
                # show only if at least one is significant
                star = " " + "/".join(s for s in stars_list if s != "ns") or ""
            ax.text(j, i, f"{val:.1f}{star}",
                    ha="center", va="center", fontsize=10,
                    color="black" if 0.2 < (val - matrix.min()) / (matrix.max() - matrix.min() + 1e-9) < 0.8 else "white")

    # Highlight diagonal (home cells)
    for k in range(len(SPECIALIST_KEYS)):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor="gold", linewidth=3))

    plt.tight_layout()
    path = join(out_dir, "heatmap_mean.png")
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2 — Box plots (one subplot per environment)
# ---------------------------------------------------------------------------

def plot_boxplots(scores: dict, stat_results: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for ax, env_key in zip(axes, ENV_KEYS):
        data   = [scores[sp][env_key] for sp in SPECIALIST_KEYS]
        colors = [SPECIALIST_COLORS[sp] for sp in SPECIALIST_KEYS]

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(SPECIALIST_KEYS) + 1))
        ax.set_xticklabels([SPECIALIST_LABELS[s] for s in SPECIALIST_KEYS],
                           fontsize=9, rotation=12)
        ax.set_title(f"{ENV_LABELS[env_key]} environment", fontsize=11)
        ax.set_ylabel("Fitness" if env_key == "flat" else "")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Annotate significance above home vs others
        home_idx  = SPECIALIST_KEYS.index(env_key)
        entry     = stat_results[env_key]
        y_max     = max(d.max() for d in data)
        y_range   = y_max - min(d.min() for d in data)
        offset    = y_range * 0.06

        for other_key in [s for s in SPECIALIST_KEYS if s != env_key]:
            other_idx = SPECIALIST_KEYS.index(other_key)
            p_val = entry[f"vs_{other_key}"]["p"]
            star  = sig_stars(p_val)
            x1, x2 = sorted([home_idx + 1, other_idx + 1])
            y_bracket = y_max + offset
            ax.annotate(
                "", xy=(x2, y_bracket), xytext=(x1, y_bracket),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
            )
            ax.text((x1 + x2) / 2, y_bracket + offset * 0.3, star,
                    ha="center", va="bottom", fontsize=11,
                    color="green" if star != "ns" else "grey")
            offset += y_range * 0.10

    # Legend
    legend_patches = [
        mpatches.Patch(color=SPECIALIST_COLORS[s], label=SPECIALIST_LABELS[s], alpha=0.75)
        for s in SPECIALIST_KEYS
    ]
    fig.legend(handles=legend_patches, loc="upper center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Fitness distribution per environment\n"
                 "(significance brackets: home vs each other specialist)",
                 fontsize=12, y=1.07)

    plt.tight_layout()
    path = join(out_dir, "boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3 — Bar chart with 95% CI (per environment, grouped by specialist)
# ---------------------------------------------------------------------------

def plot_bar_ci(scores: dict, out_dir: str) -> None:
    n_env  = len(ENV_KEYS)
    n_sp   = len(SPECIALIST_KEYS)
    x      = np.arange(n_env)
    width  = 0.22
    offsets = np.linspace(-(n_sp - 1) * width / 2, (n_sp - 1) * width / 2, n_sp)

    fig, ax = plt.subplots(figsize=(9, 5))

    for sp_key, offset in zip(SPECIALIST_KEYS, offsets):
        means, cis = [], []
        for env_key in ENV_KEYS:
            arr = scores[sp_key][env_key]
            n   = len(arr)
            mu  = arr.mean()
            ci  = stats.t.ppf(0.975, df=n - 1) * arr.std(ddof=1) / np.sqrt(n)
            means.append(mu)
            cis.append(ci)
        ax.bar(x + offset, means, width=width,
               color=SPECIALIST_COLORS[sp_key], alpha=0.8,
               label=SPECIALIST_LABELS[sp_key],
               yerr=cis, capsize=5,
               error_kw={"elinewidth": 1.2, "ecolor": "black"})

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS[e] for e in ENV_KEYS], fontsize=12)
    ax.set_ylabel("Mean fitness", fontsize=12)
    ax.set_title("Mean fitness ± 95% CI — specialists × environments", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = join(out_dir, "barplot_ci.png")
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 4 — Significance table
# ---------------------------------------------------------------------------

def plot_significance_table(scores: dict, stat_results: dict, out_dir: str) -> None:
    """Table: rows = environments, cols = (home vs sp1, home vs sp2)."""
    rows, row_labels = [], []

    for env_key in ENV_KEYS:
        entry  = stat_results[env_key]
        home   = entry["home"]
        others = [s for s in SPECIALIST_KEYS if s != home]
        row_data = []
        for other in others:
            info = entry[f"vs_{other}"]
            row_data.append(
                f"p={info['p']:.4f}  {sig_stars(info['p'])}"
            )
        rows.append(row_data)
        row_labels.append(f"{ENV_LABELS[env_key]}\n(home: {SPECIALIST_LABELS[home]})")

    other_headers = []
    for env_key in ENV_KEYS:
        others = [SPECIALIST_LABELS[s] for s in SPECIALIST_KEYS if s != env_key]
        # headers are the same for all rows (flat columns = other specialists for each env)
        other_headers = others
        break  # just need once; same structure

    col_labels = [f"vs {h}" for h in other_headers]

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.4, 2.0)

    # Color significant cells green, ns grey
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#DDDDDD")
        else:
            text = cell.get_text().get_text()
            if "***" in text or "**" in text or "*" in text:
                cell.set_facecolor("#C8E6C9")
            else:
                cell.set_facecolor("#FFCDD2")

    ax.set_title("One-sided Mann-Whitney U test: home specialist > other\n"
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
    print("\n" + "=" * 68)
    print(f"{'CROSS-EVALUATION SUMMARY':^68}")
    print("=" * 68)

    header = f"{'':22}" + "".join(f"{ENV_LABELS[e]:>12}" for e in ENV_KEYS)
    print(header)
    print("-" * 68)
    for sp in SPECIALIST_KEYS:
        row = f"{SPECIALIST_LABELS[sp]:22}"
        for env in ENV_KEYS:
            arr  = scores[sp][env]
            mark = " ✓" if sp == env else "  "
            row += f"{arr.mean():>10.1f}{mark}"
        print(row)

    print("\nStatistical tests (one-sided MWU, home > other):")
    print("-" * 68)
    for env_key in ENV_KEYS:
        entry = stat_results[env_key]
        home  = entry["home"]
        others = [s for s in SPECIALIST_KEYS if s != home]
        for other in others:
            info = entry[f"vs_{other}"]
            result = "SIGNIFICANT" if info["sig"] else "not significant"
            print(f"  {ENV_LABELS[env_key]:5s}  {SPECIALIST_LABELS[home]:20s} > "
                  f"{SPECIALIST_LABELS[other]:20s}  "
                  f"p={info['p']:.4f}  {sig_stars(info['p'])}  [{result}]")
    print("=" * 68)


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
        f"{sp}_{env}": scores[sp][env]
        for sp in SPECIALIST_KEYS for env in ENV_KEYS
    })
    print(f"  Raw scores saved: {join(OUT_DIR, 'results.npz')}")

    stat_results = run_stats(scores)
    print_summary(scores, stat_results)

    print("\nGenerating plots...")
    plot_heatmap(scores, stat_results, OUT_DIR)
    plot_boxplots(scores, stat_results, OUT_DIR)
    plot_bar_ci(scores, OUT_DIR)
    plot_significance_table(scores, stat_results, OUT_DIR)

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flat_dir", type=str,
        default=join(ROOT_DIR, "results/randomized_generalist/IT_001/990"),
        help="Directory containing flat specialist x_best.npy",
    )
    parser.add_argument(
        "--ice_dir", type=str,
        default=join(ROOT_DIR, "results/randomized_generalist/IT_002/final"),
        help="Directory containing ice specialist x_best.npy",
    )
    parser.add_argument(
        "--hill_dir", type=str,
        default=join(ROOT_DIR, "warm_start/hill_specialist_cmaes_long_2/800"),
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
