"""
trajectories.py — Run N simulations with a fixed genome (no perturbation)
and save each (x, y) trajectory to results/Before.

Usage:
    # warm start, no remap
    python trajectories.py --warm_start_dir results/flat_specialist_cmaes_long/final

    # warm start + challenge-1 joint-order remap
    python trajectories.py --warm_start_dir results/flat_specialist_cmaes_long/final --remap

    # change output folder or number of sims
    python trajectories.py --warm_start_dir ... --out_dir results/Before --n_sims 100 --n_steps 1500
"""

import argparse
import os
from os.path import join

import gymnasium as gym
import numpy as np

import evorob.world  # noqa: F401 — registers FlatEnv-v0
from evorob.utils.filesys import get_project_root
from final_project_train import FinalWorld

# re-use helpers from the training script
from train_flat_specialist import (
    N_WEIGHTS,
    FlatSpecialistWorld,
    _load_best_flat_body,
    remap_challenge1_weights,
)

ROOT_DIR   = get_project_root()
N_SIMS     = 100
N_STEPS    = 1500
OUT_DIR    = join(ROOT_DIR, "results", "Before")


def load_genome(warm_start_dir: str, apply_remap: bool) -> np.ndarray:
    ctrl_path = join(warm_start_dir, "x_best.npy")
    if not os.path.isfile(ctrl_path):
        raise FileNotFoundError(f"x_best.npy not found in {warm_start_dir}")
    genome = np.load(ctrl_path)[:N_WEIGHTS]
    if apply_remap:
        genome = remap_challenge1_weights(genome)
        print(f"  remap: Challenge-1 joint order applied")
    print(f"  genome loaded from {ctrl_path}  shape={genome.shape}")
    return genome


def run_simulation(world: FlatSpecialistWorld, genome: np.ndarray,
                   n_steps: int, seed: int) -> np.ndarray:
    """Run one episode and return trajectory of shape (T, 2)."""
    body = _load_best_flat_body()
    genome_full = np.concatenate([genome[:N_WEIGHTS], body])
    world.update_robot_xml(genome_full)

    env = gym.make("FlatEnv-v0", robot_path=world.flat_world_file,
                   render_mode=None, max_episode_steps=n_steps)
    world.controller.reset_controller(batch_size=1)
    obs, _ = env.reset(seed=seed)

    trajectory = []
    for _ in range(n_steps):
        qpos = env.unwrapped.data.qpos
        trajectory.append((float(qpos[0]), float(qpos[1])))
        action = world.controller.get_action(obs)
        if action.ndim > 1:
            action = action.squeeze(0)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()
    return np.array(trajectory)


def main(warm_start_dir: str, apply_remap: bool, out_dir: str,
         n_sims: int, n_steps: int) -> None:

    os.makedirs(out_dir, exist_ok=True)
    genome = load_genome(warm_start_dir, apply_remap)
    world  = FlatSpecialistWorld()

    print(f"\nRunning {n_sims} simulations  →  {out_dir}")
    print(f"  n_steps={n_steps}  remap={apply_remap}\n")

    for i in range(n_sims):
        seed = i  # each sim gets a distinct, reproducible seed
        traj = run_simulation(world, genome, n_steps, seed)
        out_path = join(out_dir, f"trajectory_{i:04d}.npy")
        np.save(out_path, traj)
        x_final = traj[-1, 0]
        print(f"  [{i+1:3d}/{n_sims}]  seed={seed}  steps={len(traj)}  x_final={x_final:.2f}  → {out_path}")

    print(f"\nDone. {n_sims} trajectories saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm_start_dir", type=str,
                        default=join(ROOT_DIR, "results", "flat_specialist_cmaes_long", "final"),
                        help="Directory containing x_best.npy")
    parser.add_argument("--remap", action="store_true",
                        help="Apply Challenge-1 → Final Project joint-order remap")
    parser.add_argument("--out_dir",  type=str, default=OUT_DIR)
    parser.add_argument("--n_sims",   type=int, default=N_SIMS)
    parser.add_argument("--n_steps",  type=int, default=N_STEPS)
    args = parser.parse_args()

    main(
        warm_start_dir="results/randomized_generalist/IT_001/900",
        # warm_start_dir="warm_start",
        apply_remap=False,
        # apply_remap=True,
        out_dir="After",
        # out_dir="Before",
        n_sims=500,
        n_steps=1000,
    )
