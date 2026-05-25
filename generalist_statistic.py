"""
generalist_statistic.py — Generalist vs. specialists on EvalEnv.

Evaluates flat_specialist, ice_specialist, hill_specialist, and the generalist
— each genome unchanged — on EvalEnv-v0.
Runs N_RUNS independent episodes per agent and tests whether the generalist
performs significantly better than each specialist (one-sided Mann-Whitney U
test, generalist > specialist, α=0.05).

Outputs (saved to results/generalist_stats/):
  • boxplot.png             — box plots for all 4 agents on EvalEnv
  • barplot_ci.png          — bar chart with 95% CI
  • significance_table.png  — p-value annotation table
  • results.npz             — raw scores for later analysis

Usage:
    python generalist_statistic.py \
        --flat_dir        results/flat_specialist_cmaes_long/final \
        --ice_dir         results/ice_specialist_cmaes_short/final \
        --hill_dir        results/Hill_specialist_cmaes_long/final \
        --generalist_dir  results/randomized_generalist/IT_01/final
"""

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
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
N_RUNS    = 20
N_STEPS   = 1500
ALPHA     = 0.05

AGENT_KEYS = ["flat", "ice", "hill", "generalist"]

AGENT_COLORS = {
    "flat":        "#4C72B0",
    "ice":         "#55A868",
    "hill":        "#C44E52",
    "generalist":  "#FF9800",
}
AGENT_LABELS = {
    "flat":        "Flat specialist",
    "ice":         "Ice specialist",
    "hill":        "Hill specialist",
    "generalist":  "Generalist",
}

OUT_DIR = join(ROOT_DIR, "results", "generalist_stats")


# ---------------------------------------------------------------------------
# Body params (standard ant morphology used for specialists)
# ---------------------------------------------------------------------------

def _standard_body() -> np.ndarray:
    return np.array([0.05, 0.9, 0.05, 0.9])


# ---------------------------------------------------------------------------
# Genome loading
# ---------------------------------------------------------------------------

def load_genome(agent_dir: str) -> np.ndarray:
    path = join(agent_dir, "x_best.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"x_best.npy not found in {agent_dir}")
    genome = np.load(path)[:N_WEIGHTS]
    print(f"  Loaded genome from {path}  shape={genome.shape}")
    return genome


_ASSETS = join(ROOT_DIR, "evorob", "world", "robot", "assets")


def _setup_generalist(world: FinalWorld, checkpoint_dir: str) -> None:
    """Set up a FinalWorld for the generalist using the saved Robot.xml.

    The checkpoint only stores controller weights in x_best.npy; the body
    morphology is in Robot.xml.  This function injects that XML into the
    eval terrain template instead of rebuilding the body from a genome.
    """
    robot_xml_src = join(checkpoint_dir, "Robot.xml")
    if not os.path.isfile(robot_xml_src):
        raise FileNotFoundError(f"Robot.xml not found in {checkpoint_dir}")

    # Copy body XML to temp dir so the terrain file can include it
    shutil.copy2(robot_xml_src, join(world.temp_dir.name, "Robot.xml"))

    # Load controller weights
    genome = np.load(join(checkpoint_dir, "x_best.npy"))
    world.controller.geno2pheno(genome[:world.n_weights])

    # Build eval_world_file: inject Robot.xml into eval_terrain.xml template
    tree = ET.parse(join(_ASSETS, "eval_terrain.xml"))
    root = tree.getroot()
    root.append(ET.Element("include", attrib={"file": "Robot.xml"}))
    with open(world.eval_world_file, "w") as f:
        f.write(ET.tostring(root, encoding="unicode"))

    # EvalEnv needs the hilly_hfield image; copy from assets if not present
    hfield_dst = join(world.temp_dir.name, "hilly_hfield.png")
    if not os.path.isfile(hfield_dst):
        shutil.copy2(join(_ASSETS, "hilly_hfield.png"), hfield_dst)


# ---------------------------------------------------------------------------
# Single-episode evaluation on EvalEnv
# ---------------------------------------------------------------------------

def evaluate_episode(world: FinalWorld) -> tuple[float, float]:
    """Run one episode on EvalEnv-v0; return (total_reward, distance_traveled)."""
    import gymnasium as gym
    env = gym.make("EvalEnv-v0", robot_path=world.eval_world_file,
                   max_episode_steps=N_STEPS)
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
# Video recording
# ---------------------------------------------------------------------------

def record_video(world: FinalWorld, ag_key: str, out_dir: str) -> None:
    """Record one episode on EvalEnv-v0 and save to out_dir/video_{ag_key}.mp4."""
    try:
        import imageio
        import gymnasium as gym
        env = gym.make("EvalEnv-v0", robot_path=world.eval_world_file,
                       render_mode="rgb_array", max_episode_steps=N_STEPS)
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            frames.append(env.render())
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()
        path = join(out_dir, f"video_{ag_key}.mp4")
        imageio.mimwrite(path, frames, fps=20)
        print(f"  Video: {path}")
    except Exception as exc:
        print(f"  Video skipped ({ag_key}): {exc}")


# ---------------------------------------------------------------------------
# Full evaluation — all agents on EvalEnv
# ---------------------------------------------------------------------------

def run_all(genomes: dict[str, np.ndarray], generalist_dir: str) -> dict[str, dict[str, np.ndarray]]:
    """
    Returns scores[agent_key] = {"reward": np.ndarray, "distance": np.ndarray},
    each of shape (N_RUNS,).
    """
    scores: dict[str, dict[str, list]] = {
        k: {"reward": [], "distance": []} for k in AGENT_KEYS
    }

    for ag_key in AGENT_KEYS:
        print(f"\n── Evaluating {AGENT_LABELS[ag_key]} on EvalEnv ──")
        world = FinalWorld()

        if ag_key == "generalist":
            _setup_generalist(world, generalist_dir)
        else:
            genome_full = np.concatenate([genomes[ag_key], _standard_body()])
            world.update_robot_xml(genome_full)

        print(f"   ", end="", flush=True)
        for _ in range(N_RUNS):
            reward, distance = evaluate_episode(world)
            scores[ag_key]["reward"].append(reward)
            scores[ag_key]["distance"].append(distance)
            print(".", end="", flush=True)
        mean_r = np.mean(scores[ag_key]["reward"])
        mean_d = np.mean(scores[ag_key]["distance"])
        print(f"  reward={mean_r:.1f}  distance={mean_d:.1f}")

        record_video(world, ag_key, OUT_DIR)
        world.temp_dir.cleanup()

    return {
        k: {"reward": np.array(v["reward"]), "distance": np.array(v["distance"])}
        for k, v in scores.items()
    }


# ---------------------------------------------------------------------------
# Statistical tests — generalist > each specialist
# ---------------------------------------------------------------------------

def run_stats(scores: dict) -> dict:
    """
    For each specialist, run two one-sided Mann-Whitney U tests:
      generalist > specialist on reward
      generalist > specialist on distance

    Returns results[specialist_key] = {
        "reward":   {"stat", "p", "sig"},
        "distance": {"stat", "p", "sig"},
    }
    """
    gen_r = scores["generalist"]["reward"]
    gen_d = scores["generalist"]["distance"]
    results = {}
    for sp_key in ["flat", "ice", "hill"]:
        stat_r, p_r = stats.mannwhitneyu(gen_r, scores[sp_key]["reward"],   alternative="greater")
        stat_d, p_d = stats.mannwhitneyu(gen_d, scores[sp_key]["distance"], alternative="greater")
        results[sp_key] = {
            "reward":   {"stat": stat_r, "p": p_r, "sig": p_r < ALPHA},
            "distance": {"stat": stat_d, "p": p_d, "sig": p_d < ALPHA},
        }
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
# Shared bracket annotation helper
# ---------------------------------------------------------------------------

def _add_brackets(ax, stat_results: dict, metric: str, y_max: float, y_range: float) -> None:
    gen_idx = AGENT_KEYS.index("generalist") + 1
    offset  = y_range * 0.06
    for sp_key in ["flat", "ice", "hill"]:
        sp_idx    = AGENT_KEYS.index(sp_key) + 1
        p_val     = stat_results[sp_key][metric]["p"]
        star      = sig_stars(p_val)
        x1, x2   = sorted([sp_idx, gen_idx])
        y_bracket = y_max + offset
        ax.annotate("", xy=(x2, y_bracket), xytext=(x1, y_bracket),
                    arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
        ax.text((x1 + x2) / 2, y_bracket + offset * 0.3, star,
                ha="center", va="bottom", fontsize=12,
                color="green" if star != "ns" else "grey")
        offset += y_range * 0.10


# ---------------------------------------------------------------------------
# Plot 1 — Box plots (reward | distance), side by side
# ---------------------------------------------------------------------------

def plot_boxplot(scores: dict, stat_results: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = [AGENT_COLORS[k] for k in AGENT_KEYS]

    for ax, metric, ylabel, title in [
        (axes[0], "reward",   "Total reward (EvalEnv)",     "Reward"),
        (axes[1], "distance", "Distance traveled (EvalEnv)", "Distance"),
    ]:
        data = [scores[k][metric] for k in AGENT_KEYS]
        bp   = ax.boxplot(data, patch_artist=True, notch=False,
                          medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(AGENT_KEYS) + 1))
        ax.set_xticklabels([AGENT_LABELS[k] for k in AGENT_KEYS], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        y_max   = max(d.max() for d in data)
        y_range = y_max - min(d.min() for d in data)
        _add_brackets(ax, stat_results, metric, y_max, y_range)

    legend_patches = [
        mpatches.Patch(color=AGENT_COLORS[k], label=AGENT_LABELS[k], alpha=0.75)
        for k in AGENT_KEYS
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Generalist vs. specialists on EvalEnv\n"
                 "(brackets: generalist > specialist, one-sided MWU)",
                 fontsize=12, y=1.07)

    plt.tight_layout()
    path = join(out_dir, "boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2 — Bar chart with 95% CI (reward | distance), side by side
# ---------------------------------------------------------------------------

def plot_bar_ci(scores: dict, stat_results: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x      = np.arange(len(AGENT_KEYS))
    colors = [AGENT_COLORS[k] for k in AGENT_KEYS]

    for ax, metric, ylabel, title in [
        (axes[0], "reward",   "Mean total reward ± 95% CI",     "Reward"),
        (axes[1], "distance", "Mean distance traveled ± 95% CI", "Distance"),
    ]:
        means, cis = [], []
        for k in AGENT_KEYS:
            arr = scores[k][metric]
            n   = len(arr)
            mu  = arr.mean()
            ci  = stats.t.ppf(0.975, df=n - 1) * arr.std(ddof=1) / np.sqrt(n)
            means.append(mu)
            cis.append(ci)

        ax.bar(x, means, color=colors, alpha=0.8,
               yerr=cis, capsize=6,
               error_kw={"elinewidth": 1.5, "ecolor": "black"})

        gen_x   = x[AGENT_KEYS.index("generalist")]
        y_top   = max(m + c for m, c in zip(means, cis))
        y_range = y_top - min(means)
        offset  = y_range * 0.07
        for sp_key in ["flat", "ice", "hill"]:
            sp_x  = x[AGENT_KEYS.index(sp_key)]
            p_val = stat_results[sp_key][metric]["p"]
            star  = sig_stars(p_val)
            xm    = (sp_x + gen_x) / 2
            y_b   = y_top + offset
            ax.annotate("", xy=(max(sp_x, gen_x), y_b),
                        xytext=(min(sp_x, gen_x), y_b),
                        arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
            ax.text(xm, y_b + offset * 0.2, star,
                    ha="center", va="bottom", fontsize=12,
                    color="green" if star != "ns" else "grey")
            offset += y_range * 0.12

        ax.set_xticks(x)
        ax.set_xticklabels([AGENT_LABELS[k] for k in AGENT_KEYS], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    legend_patches = [
        mpatches.Patch(color=AGENT_COLORS[k], label=AGENT_LABELS[k], alpha=0.75)
        for k in AGENT_KEYS
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Generalist vs. specialists on EvalEnv", fontsize=12, y=1.07)

    plt.tight_layout()
    path = join(out_dir, "barplot_ci.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3 — Significance table (reward + distance columns)
# ---------------------------------------------------------------------------

def plot_significance_table(stat_results: dict, out_dir: str) -> None:
    rows       = []
    row_labels = []
    for sp_key in ["flat", "ice", "hill"]:
        r = stat_results[sp_key]["reward"]
        d = stat_results[sp_key]["distance"]
        rows.append([
            f"p={r['p']:.4f}  {sig_stars(r['p'])}",
            f"p={d['p']:.4f}  {sig_stars(d['p'])}",
        ])
        row_labels.append(f"Generalist > {AGENT_LABELS[sp_key]}")

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=["Reward (MWU)", "Distance (MWU)"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.4, 2.2)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#DDDDDD")
        else:
            text = cell.get_text().get_text()
            if "***" in text or "**" in text or "*" in text:
                cell.set_facecolor("#C8E6C9")
            else:
                cell.set_facecolor("#FFCDD2")

    ax.set_title("One-sided MWU: generalist > specialist on EvalEnv\n"
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
    print("\n" + "=" * 72)
    print(f"{'GENERALIST vs. SPECIALISTS — EvalEnv':^72}")
    print("=" * 72)
    print(f"{'Agent':25}  {'Reward mean':>12}  {'Dist mean':>10}  {'Dist std':>9}")
    print("-" * 72)
    for k in AGENT_KEYS:
        r = scores[k]["reward"]
        d = scores[k]["distance"]
        print(f"{AGENT_LABELS[k]:25}  {r.mean():12.1f}  {d.mean():10.1f}  {d.std():9.1f}")

    for metric in ["reward", "distance"]:
        label = "Reward" if metric == "reward" else "Distance"
        print(f"\nStatistical tests on {label} (one-sided MWU, generalist > specialist):")
        print("-" * 72)
        for sp_key in ["flat", "ice", "hill"]:
            info   = stat_results[sp_key][metric]
            result = "SIGNIFICANT" if info["sig"] else "not significant"
            print(f"  Generalist > {AGENT_LABELS[sp_key]:20s}  "
                  f"p={info['p']:.4f}  {sig_stars(info['p'])}  [{result}]")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(flat_dir: str, ice_dir: str, hill_dir: str, generalist_dir: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nLoading genomes...")
    genomes = {
        "flat":  load_genome(flat_dir),
        "ice":   load_genome(ice_dir),
        "hill":  load_genome(hill_dir),
    }

    print(f"\nEvaluating {N_RUNS} runs × 4 agents on EvalEnv "
          f"({N_RUNS * 4} total episodes, {N_STEPS} steps each)...")
    scores = run_all(genomes, generalist_dir)

    np.savez(join(OUT_DIR, "results.npz"), **{
        f"{k}_{metric}": scores[k][metric]
        for k in AGENT_KEYS for metric in ("reward", "distance")
    })
    print(f"  Raw scores saved: {join(OUT_DIR, 'results.npz')}")

    stat_results = run_stats(scores)
    print_summary(scores, stat_results)

    print("\nGenerating plots...")
    plot_boxplot(scores, stat_results, OUT_DIR)
    plot_bar_ci(scores, stat_results, OUT_DIR)
    plot_significance_table(stat_results, OUT_DIR)

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
    parser.add_argument(
        "--generalist_dir", type=str,
        default=join(ROOT_DIR, "results/randomized_generalist/IT_01/575"),
        help="Directory containing generalist x_best.npy",
    )
    parser.add_argument("--n_runs",  type=int, default=N_RUNS,
                        help="Independent episodes per agent")
    parser.add_argument("--n_steps", type=int, default=N_STEPS,
                        help="Steps per episode")
    args = parser.parse_args()

    N_RUNS  = args.n_runs
    N_STEPS = args.n_steps

    main(
        flat_dir=args.flat_dir,
        ice_dir=args.ice_dir,
        hill_dir=args.hill_dir,
        generalist_dir=args.generalist_dir,
    )
