"""
MICRO-515 Final Project — Multi-task Robot Evolution
=====================================================
Evolve a legged robot (body + controller) to walk in the +x direction across
three training environments simultaneously.  The genotype encodes both the
neural controller weights and the body morphology (leg lengths).

Training environments (3 objectives)
-------------------------------------
1  Flat  — standard ground, good friction  (FlatEnv-v0  / flat_world.xml)
2  Ice   — slippery ground, low friction   (IceEnv-v0   / ice_world.xml)
3  Hill  — procedural hilly terrain        (HillEnv-v0  / hill_world.xml)

The evaluation terrain is separate and fixed.  Students test their best
evolved robot on it using final_project_test.py — it is not trained on.

Genotype layout
---------------
  genotype[:n_weights]  → controller (560 params for MLP 27→16→8)
  genotype[n_weights:]  → body (4 params, diagonal symmetry)

x_best.npy always stores only the 560 controller params.
Body params of the best individual are stored in x_best_body.npy.
"""

import copy
import os
import shutil
import time
import xml.etree.ElementTree as xml
from os.path import join
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import scipy.ndimage
from PIL import Image
from gymnasium.vector import AsyncVectorEnv

import evorob.world                         # registers EvalEnv-v0
from best_genome_selection import (
    build_diverse_initial_population,
    load_top_k_flat, load_top_k_hill, load_top_k_ice,
)
from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root
from evorob.world.base import World
from evorob.world.robot.controllers.mlp import NeuralNetworkController
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

ROOT_DIR = get_project_root()
_ASSETS  = join(ROOT_DIR, "evorob", "world", "robot", "assets")
MAX_EPISODE_STEPS = 1000  # fixed for leaderboard — do not change


# ---------------------------------------------------------------------------
# FinalWorld — body + brain co-evolution across multiple terrains
# ---------------------------------------------------------------------------

class FinalWorld(World):
    """Translates a genotype into a robot phenotype and evaluates it.

    The genotype is a 1-D array: [controller_params | body_params].
    Body uses diagonal symmetry: 4 params encode 8 segment lengths.
      front-left = back-right  (front_leg, front_ankle)
      front-right = back-left  (back_leg,  back_ankle)
    """

    def __init__(self):
        self.controller = NeuralNetworkController(
            input_size=27, output_size=8, hidden_size=16
        )

        self.n_weights     = self.controller.n_params  # 560
        self.n_body_params = 4   # diagonal symmetry: front_leg, front_ankle, back_leg, back_ankle
        self.n_params      = self.n_weights + self.n_body_params  # 564

        self.temp_dir        = TemporaryDirectory()
        self.flat_world_file = join(self.temp_dir.name, "WorldFlat.xml")
        self.ice_world_file  = join(self.temp_dir.name, "WorldIce.xml")
        self.hill_world_file = join(self.temp_dir.name, "WorldHill.xml")
        self.world_file      = self.hill_world_file

        self.joint_limits = [
            [-30, 30], [30, 70],
            [-30, 30], [-70, -30],
            [-30, 30], [-70, -30],
            [-30, 30], [30, 70],
        ]
        self.joint_axis = [
            [0, 0, 1], [-1, 1, 0],
            [0, 0, 1], [1, 1, 0],
            [0, 0, 1], [-1, 1, 0],
            [0, 0, 1], [1, 1, 0],
        ]

        self.sensor_fn = None

        self._create_terrain_file("terrain.png")

        # Cache parsed XML templates — re-used every individual via deepcopy
        self._template_roots = {
            "flat": xml.parse(join(_ASSETS, "flat_world.xml")).getroot(),
            "ice":  xml.parse(join(_ASSETS, "ice_world.xml")).getroot(),
            "hill": xml.parse(join(_ASSETS, "hill_world.xml")).getroot(),
        }

    # ------------------------------------------------------------------
    # Genotype → phenotype
    # ------------------------------------------------------------------

    def geno2pheno(self, genotype: np.ndarray):
        """Decode genotype into controller weights and body parameters.

        Body uses bilateral symmetry (4 params → 8 segment lengths):
        front-left leg/ankle = front-right leg/ankle  (body_params[0:2])
        back-left leg/ankle  = back-right leg/ankle   (body_params[2:4])

        If genotype has only n_weights entries (no body params), a neutral
        body (all zeros → 0.35 m per segment) is used.
        """
        control_params = genotype[:self.n_weights]
        body_raw = (genotype[self.n_weights:] if len(genotype) > self.n_weights
                    else np.zeros(self.n_body_params))
        body_params = (body_raw + 1) / 4 + 0.1
        self.controller.geno2pheno(control_params)

        front_leg, front_ankle, back_leg, back_ankle = body_params

        # Bilateral symmetry
        front_left_leg    = front_right_leg    = front_leg
        front_left_ankle  = front_right_ankle  = front_ankle
        back_left_leg     = back_right_leg     = back_leg
        back_left_ankle   = back_right_ankle   = back_ankle

        front_left_hip_xyz   = np.array([0.2, 0.2, 0])
        front_left_knee_xyz  = np.array([np.sqrt(0.5 * front_left_leg ** 2),
                                        np.sqrt(0.5 * front_left_leg ** 2), 0]) + front_left_hip_xyz
        front_left_toe_xyz   = np.array([np.sqrt(0.5 * front_left_ankle ** 2),
                                        np.sqrt(0.5 * front_left_ankle ** 2), 0]) + front_left_knee_xyz

        front_right_hip_xyz  = np.array([-0.2, 0.2, 0])
        front_right_knee_xyz = np.array([-np.sqrt(0.5 * front_right_leg ** 2),
                                        np.sqrt(0.5 * front_right_leg ** 2), 0]) + front_right_hip_xyz
        front_right_toe_xyz  = np.array([-np.sqrt(0.5 * front_right_ankle ** 2),
                                        np.sqrt(0.5 * front_right_ankle ** 2), 0]) + front_right_knee_xyz

        back_left_hip_xyz    = np.array([-0.2, -0.2, 0])
        back_left_knee_xyz   = np.array([-np.sqrt(0.5 * back_left_leg ** 2),
                                        -np.sqrt(0.5 * back_left_leg ** 2), 0]) + back_left_hip_xyz
        back_left_toe_xyz    = np.array([-np.sqrt(0.5 * back_left_ankle ** 2),
                                        -np.sqrt(0.5 * back_left_ankle ** 2), 0]) + back_left_knee_xyz

        back_right_hip_xyz   = np.array([0.2, -0.2, 0])
        back_right_knee_xyz  = np.array([np.sqrt(0.5 * back_right_leg ** 2),
                                        -np.sqrt(0.5 * back_right_leg ** 2), 0]) + back_right_hip_xyz
        back_right_toe_xyz   = np.array([np.sqrt(0.5 * back_right_ankle ** 2),
                                        -np.sqrt(0.5 * back_right_ankle ** 2), 0]) + back_right_knee_xyz

        points = np.vstack([front_left_hip_xyz,  front_left_knee_xyz,  front_left_toe_xyz,
                            front_right_hip_xyz, front_right_knee_xyz, front_right_toe_xyz,
                            back_left_hip_xyz,   back_left_knee_xyz,   back_left_toe_xyz,
                            back_right_hip_xyz,  back_right_knee_xyz,  back_right_toe_xyz])

        connectivity_mat = np.array(
            [[150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )
        return points, connectivity_mat

        # ------------------------------------------------------------------
        # Robot XML generation
        # ------------------------------------------------------------------

    def update_robot_xml(self, genotype: np.ndarray) -> None:
        """Build robot body XML from genotype and inject into every terrain template."""
        points, connectivity_mat = self.geno2pheno(genotype)
        robot = AntRobot(
            points, connectivity_mat, self.joint_limits, self.joint_axis,
            name="Robot", verbose=False,
        )
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)

        for name, world_file in [
            ("flat", self.flat_world_file),
            ("ice",  self.ice_world_file),
            ("hill", self.hill_world_file),
        ]:
            root = copy.deepcopy(self._template_roots[name])
            root.append(xml.Element("include", attrib={"file": "Robot.xml"}))
            with open(world_file, "w") as f:
                f.write(xml.tostring(root, encoding="unicode"))

    def _create_terrain_file(self, filename: str, width: int = 200, depth: int = 400):
        """Hill terrain PNG: smooth start, bumpy middle, smooth end."""
        slope_deg  = 5.0
        bump_scale = 0.08
        sigma      = 4.0

        rise = np.tan(np.deg2rad(slope_deg))
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)

        slope_map = np.clip(X * rise, 0, 1)

        rng   = np.random.default_rng(42)
        noise = rng.uniform(0, 1, (width, depth))
        bump_envelope = np.sin(np.pi * X)
        noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * bump_envelope
        noise_map = noise * bump_scale

        terrain = np.clip(slope_map + noise_map, 0, 1)
        terrain[-1, -1] = 1

        img = Image.fromarray((terrain * 255).astype(np.uint8), mode="L")
        img.save(join(self.temp_dir.name, filename))

    # ------------------------------------------------------------------
    # Per-terrain evaluation
    # ------------------------------------------------------------------

    def _run_env(self, env_id: str, world_file: str, n_repeats: int, n_steps: int) -> float:
        """Run n_repeats parallel episodes and return the mean total reward."""
        envs = AsyncVectorEnv([
            (lambda eid, wf: lambda: gym.make(
                eid, robot_path=wf, max_episode_steps=n_steps
            ))(env_id, world_file)
            for _ in range(n_repeats)
        ])
        self.controller.reset_controller(batch_size=n_repeats)
        rewards = np.zeros((n_steps, n_repeats))
        obs, _ = envs.reset()
        if self.sensor_fn is not None:
            obs = self.sensor_fn(obs)
        done = np.zeros(n_repeats, dtype=bool)
        for t in range(n_steps):
            actions = np.where(done[:, None], 0, self.controller.get_action(obs))
            obs, r, terminated, truncated, _ = envs.step(actions)
            if self.sensor_fn is not None:
                obs = self.sensor_fn(obs)
            rewards[t, ~done] = r[~done]
            done |= terminated | truncated
            if done.all():
                break
        envs.close()
        return float(rewards.sum(axis=0).mean())

    def _eval_flat(self, n_repeats: int = 4, n_steps: int = 500) -> float:
        return self._run_env("FlatEnv-v0", self.flat_world_file, n_repeats, n_steps)

    def _eval_ice(self, n_repeats: int = 4, n_steps: int = 500) -> float:
        return self._run_env("IceEnv-v0", self.ice_world_file, n_repeats, n_steps)

    def _eval_hill(self, n_repeats: int = 4, n_steps: int = 500) -> float:
        return self._run_env("HillEnv-v0", self.hill_world_file, n_repeats, n_steps)

    def create_env(self, render_mode: str = "rgb_array", **kwargs):
        return gym.make("HillEnv-v0", robot_path=self.hill_world_file,
                        render_mode=render_mode, **kwargs)

    # ------------------------------------------------------------------
    # Combined fitness for NSGA-II
    # ------------------------------------------------------------------

    def evaluate_individual(self, genotype: np.ndarray,
                            n_repeats: int = 4, n_steps: int = 500) -> np.ndarray:
        """Evaluate one genotype on all three training environments.

        Returns a 1-D array of three objective values: [flat, ice, hill].
        """
        self.update_robot_xml(genotype)
        # Sequential evaluation: forking AsyncVectorEnv from inside ThreadPoolExecutor
        # threads deadlocks on Linux (fork() copies only the calling thread but
        # inherits mutexes held by other threads).
        results = [
            self._eval_flat(n_repeats, n_steps),
            self._eval_ice(n_repeats, n_steps),
            self._eval_hill(n_repeats, n_steps),
        ]
        return np.array(results)


# ---------------------------------------------------------------------------
# Neutral leaderboard evaluation  (TA-graded — do not modify)
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
    n_episodes: int = 256,
) -> dict | None:
    """Evaluate the best genotype from a checkpoint on all three training terrains.

    Loads x_best.npy (560 controller params) and x_best_body.npy (4 body params,
    if available).  Falls back to a neutral body if x_best_body.npy is absent.

    Args:
        checkpoint_dir: Path to your NSGA-II checkpoint folder.
        output_dir:     Where to save the score file and videos.
        n_episodes:     Episodes per terrain (256 for submission).
    """
    MAX_STEPS = MAX_EPISODE_STEPS   # DO NOT CHANGE
    SEED      = 0                   # DO NOT CHANGE

    last_gen = get_last_checkpoint_dir(checkpoint_dir)

    def _load(fname):
        for d in ([last_gen] if last_gen else []) + [checkpoint_dir]:
            p = join(d, fname)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        return None

    x_best = _load("x_best.npy")
    if x_best is None:
        print(f"ERROR: x_best.npy not found in '{checkpoint_dir}'.")
        return None
    print(f"Loaded x_best  (shape: {x_best.shape})")

    world = FinalWorld()

    # Handle 560-dim x_best (controller only) — load body params or use neutral
    if x_best.shape[0] == world.n_weights:
        x_best_body = _load("x_best_body.npy")
        if x_best_body is not None:
            print(f"Loaded x_best_body  (shape: {x_best_body.shape})")
        else:
            x_best_body = np.zeros(world.n_body_params)
            print("x_best_body.npy not found — using neutral body (0.35 m per segment).")
        x_best = np.concatenate([x_best, x_best_body])

    world.update_robot_xml(x_best)
    ctrl_name = type(world.controller).__name__
    print(f"Controller: {ctrl_name}  |  n_weights={world.n_weights}"
          f"  |  genotype size={world.n_params}\n")

    terrains = {
        "flat": ("FlatEnv-v0", world.flat_world_file),
        "ice":  ("IceEnv-v0",  world.ice_world_file),
        "hill": ("HillEnv-v0", world.hill_world_file),
    }

    def _neutral(info: dict) -> float:
        return (float(info.get("healthy_reward", 1.0))
                + float(info.get("x_position",   0.0))
                - float(info.get("ctrl_cost",     0.0))
                - float(info.get("cfrc_cost",     0.0)))

    def _stats(values: list) -> dict:
        arr = np.asarray(values)
        return dict(mean=float(arr.mean()), std=float(arr.std()),
                    best=float(arr.max()), worst=float(arr.min()), values=values)

    def _run(env_id: str, world_file: str) -> list:
        rng = np.random.default_rng(SEED)
        env = gym.make(env_id, robot_path=world_file, max_episode_steps=MAX_STEPS)
        rewards = []
        for ep in range(n_episodes):
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
            total, done = 0.0, False
            while not done:
                action = world.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, info = env.step(action)
                total += _neutral(info)
                done = terminated or truncated
            rewards.append(total)
        env.close()
        return rewards

    def _record(env_id: str, world_file: str, out_path: str) -> None:
        try:
            import imageio
            env = gym.make(env_id, robot_path=world_file,
                           render_mode="rgb_array", max_episode_steps=MAX_STEPS)
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=SEED)
            frames = []
            for _ in range(MAX_STEPS):
                frames.append(env.render())
                action = world.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            env.close()
            imageio.mimwrite(out_path, frames, fps=20)
            print(f"  Video: {out_path}")
        except Exception as exc:
            print(f"  Video skipped: {exc}")

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for terrain_name, (env_id, world_file) in terrains.items():
        print(f"  Running {terrain_name}  ({n_episodes} episodes)...", flush=True)
        results[terrain_name] = _stats(_run(env_id, world_file))

    t_names = list(results.keys())
    col_w = 12
    hdr = f"  {'Ep':>4}   " + "   ".join(f"{n.capitalize():>{col_w}}" for n in t_names)
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    for ep in range(n_episodes):
        row = f"  {ep + 1:>4}   " + "   ".join(
            f"{results[n]['values'][ep]:>{col_w}.2f}" for n in t_names
        )
        print(row)
    print(sep)
    print(f"  {'mean':>4}   " + "   ".join(
        f"{results[n]['mean']:>{col_w}.2f}" for n in t_names
    ))
    print(f"  {'std':>4}   " + "   ".join(
        f"{results[n]['std']:>{col_w}.2f}" for n in t_names
    ))
    print()

    print("Recording videos...")
    for terrain_name, (env_id, world_file) in terrains.items():
        _record(env_id, world_file, join(output_dir, f"evaluation_{terrain_name}.mp4"))

    score_path = join(output_dir, "evaluation_score.txt")
    col = 60
    with open(score_path, "w") as f:
        f.write("=" * col + "\n")
        f.write("MICRO-515 Final Project — Evaluation Results\n")
        f.write("=" * col + "\n\n")
        f.write(f"Controller      : {ctrl_name} ({world.n_weights} params)\n")
        f.write(f"Genotype size   : {world.n_params}"
                f"  (controller={world.n_weights}, body={world.n_body_params})\n")
        f.write(f"Checkpoint      : {checkpoint_dir}\n")
        f.write(f"Episodes/terrain: {n_episodes}\n")
        f.write(f"Reward          : healthy_reward + x_position - ctrl_cost - cfrc_cost\n\n")

        f.write("=" * col + "\n")
        f.write("SUMMARY\n")
        f.write("=" * col + "\n")
        f.write(f"{'Terrain':<8} {'Mean':>9} {'Std':>8} {'Best':>9} {'Worst':>9}\n")
        f.write("-" * col + "\n")
        for terrain_name, r in results.items():
            f.write(f"{terrain_name:<8} {r['mean']:9.2f} {r['std']:8.2f}"
                    f" {r['best']:9.2f} {r['worst']:9.2f}\n")
        f.write("\n")

        for terrain_name, r in results.items():
            f.write("-" * 50 + "\n")
            f.write(f"{terrain_name.upper()} — Per-episode rewards\n")
            f.write("-" * 50 + "\n")
            for i, v in enumerate(r["values"]):
                f.write(f"  Episode {i + 1:3d}: {v:10.2f}\n")
            f.write("\n")

    print(f"\nScore saved to: {score_path}")
    print("=" * col)
    for terrain_name, r in results.items():
        print(f"  {terrain_name:<6}: {r['mean']:8.2f} ± {r['std']:7.2f}"
              f"  best={r['best']:.2f}  worst={r['worst']:.2f}")
    print("=" * col)
    return results


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def build_initial_population(
    n_params: int,
    n_weights: int,
    n_body_params: int,
    bounds: tuple,
    n_random: int = 128,
    specialists: list = None,
    n_per_specialist: int = 128,
    ctrl_noise_std: float = 0.1,
    body_noise_std: float = 0.1,
    random_seed: int = 42,
) -> np.ndarray:
    """Build the initial population using random individuals + specialist warm-starts.

    Each specialist contributes n_per_specialist individuals:
      - 1 exact copy (controller weights + neutral body)
      - (n_per_specialist - 1) copies with Gaussian noise on both controller
        weights and body params

    Body params are initialised to 0 (neutral: 0.35 m per segment via the
    (g+1)/4+0.1 decode) and perturbed with body_noise_std for noisy variants.

    Args:
        n_params:         Full genotype length (560 ctrl + 4 body = 564).
        n_weights:        Controller-only length (560).
        n_body_params:    Body-only length (4).
        bounds:           (min, max) clipping range.
        n_random:         Number of fully random individuals.
        specialists:      List of (n_weights,) arrays; each is one warm-start.
        n_per_specialist: Total individuals derived from each specialist.
        ctrl_noise_std:   Std of Gaussian noise added to controller weights.
        body_noise_std:   Std of Gaussian noise added to body params.
        random_seed:      RNG seed for reproducibility.

    Returns:
        np.ndarray of shape (n_random + len(specialists)*n_per_specialist, n_params).
    """
    rng = np.random.default_rng(random_seed)
    parts = []

    # --- Fully random individuals ---
    random_pop = rng.uniform(bounds[0], bounds[1], (n_random, n_params))
    # Keep body params small at initialisation so legs start near neutral
    random_pop[:, n_weights:] = rng.normal(0, body_noise_std, (n_random, n_body_params))
    random_pop = np.clip(random_pop, bounds[0], bounds[1])
    parts.append(random_pop)

    # --- Specialist warm-starts ---
    if specialists:
        neutral_body = np.zeros(n_body_params)
        for ctrl_weights in specialists:
            group = []
            # 1 exact specialist with neutral body
            exact = np.concatenate([ctrl_weights, neutral_body])
            group.append(np.clip(exact, bounds[0], bounds[1]))

            # (n_per_specialist - 1) noisy variants
            for _ in range(n_per_specialist - 1):
                noisy_ctrl = ctrl_weights + rng.normal(0, ctrl_noise_std, n_weights)
                noisy_body = rng.normal(0, body_noise_std, n_body_params)
                noisy = np.concatenate([noisy_ctrl, noisy_body])
                group.append(np.clip(noisy, bounds[0], bounds[1]))

            parts.append(np.array(group))

    return np.vstack(parts)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def remap_challenge1_weights(weights: np.ndarray) -> np.ndarray:
    """Remap Challenge 1 output-layer weights to Final Project joint order.

    Challenge 1:    [back-right-hip, back-right-ankle, front-left-hip, front-left-ankle,
                     front-right-hip, front-right-ankle, back-left-hip, back-left-ankle]
    Final Project:  [front-left-hip, front-left-ankle, front-right-hip, front-right-ankle,
                     back-left-hip, back-left-ankle, back-right-hip, back-right-ankle]
    """
    n_input, n_hidden, n_output = 27, 16, 8
    n_con1 = n_input * n_hidden   # 432
    n_con2 = n_hidden * n_output  # 128

    input_weights  = weights[:n_con1]
    output_weights = weights[n_con1:n_con1 + n_con2].reshape(n_output, n_hidden)

    reorder = [2, 3, 4, 5, 6, 7, 0, 1]
    output_weights_reordered = output_weights[reorder, :]

    return np.concatenate([input_weights, output_weights_reordered.flatten()])


def run_multi_task_evolution(
    num_generations: int = 1000,
    population_size: int = 512,
    n_parents:       int = 256,
    n_repeats:       int = 3,
    n_steps:         int = 500,
    mutation_prob:   float = 0.2,
    crossover_prob:  float = 0.7,
    bounds:          tuple = (-10, 10),
    ckpt_interval:   int = 10,
    results_dir:     str = None,
    random_seed:     int = 42,
    flat_top_k:      np.ndarray = None,  # shape (k, 560)
    ice_top_k:       np.ndarray = None,  # shape (k, 560)
    hill_top_k:      np.ndarray = None,  # shape (k, 560)
    n_per_genome:    int = 32,
    ctrl_noise_std:  float = 0.15,
    body_noise_std:  float = 0.1,
) -> None:
    """Run NSGA-II multi-objective evolution across flat, ice, and hill terrains.

    Population initialisation
    -------------------------
    The initial population of `population_size` individuals is built as follows
    (requires population_size = n_random + n_active_terrains * k * n_per_genome):
      - 128 fully random individuals (body params near neutral via small noise)
      - For each terrain with k provided genomes:
          1 exact copy + (n_per_genome - 1) noisy variants per genome
    Body params of every specialist-derived individual are initialised with
    small Gaussian noise around neutral (0) so the EA can discover good morphologies.

    Checkpoint format
    -----------------
    x_best.npy     : 560-dim controller weights of best individual found so far.
    x_best_body.npy: 4-dim body params of that individual.
    Robot.xml      : MuJoCo XML of the best robot morphology.
    """
    np.random.seed(random_seed)

    world = FinalWorld()
    print(f"Genotype : {world.n_params} params"
          f"  (controller={world.n_weights}, body={world.n_body_params})")

    if results_dir is None:
        results_dir = join(ROOT_DIR, "results", "final_project")

    ea = NSGAII(
        population_size=population_size,
        n_opt_params=world.n_params,
        n_parents=n_parents,
        num_generations=num_generations,
        bounds=bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        output_dir=results_dir,
    )

    terrain_arrays = [flat_top_k, ice_top_k, hill_top_k]
    terrain_names  = ["flat", "ice", "hill"]
    active = [(name, arr) for name, arr in zip(terrain_names, terrain_arrays) if arr is not None]
    k = len(active[0][1]) if active else 0
    n_random = population_size - len(active) * k * n_per_genome

    if n_random < 0:
        raise ValueError(
            f"population_size ({population_size}) too small for {len(active)} terrains "
            f"× {k} genomes × {n_per_genome} = {len(active) * k * n_per_genome}.  "
            f"Need at least {len(active) * k * n_per_genome}."
        )

    print(f"Initial population: {n_random} random + "
          + " + ".join(f"{k}×{n_per_genome} {name}" for name, _ in active))

    init_pop = build_diverse_initial_population(
        n_params=world.n_params,
        n_weights=world.n_weights,
        n_body_params=world.n_body_params,
        bounds=bounds,
        n_random=n_random,
        flat_top_k=flat_top_k,
        ice_top_k=ice_top_k,
        hill_top_k=hill_top_k,
        n_per_genome=n_per_genome,
        ctrl_noise_std=ctrl_noise_std,
        body_noise_std=body_noise_std,
        random_seed=random_seed,
    )
    ea.initialise_x0 = lambda: init_pop

    n_obj = 3
    print(f"\nRunning {num_generations} generations  pop={population_size}")
    print(f"Objectives : [flat, ice, hill]")
    print(f"Checkpoints: {results_dir}\n")

    os.makedirs(results_dir, exist_ok=True)
    _best_xml_stage = join(results_dir, "_best_robot.xml")
    _best_scalar    = -np.inf

    t_run_start = time.time()
    for gen in range(num_generations):
        t_gen_start = time.time()
        pop = ea.ask()
        n_pop = len(pop)
        fitnesses = np.empty((n_pop, n_obj))
        print(f"\n[Gen {gen+1}/{num_generations}]  evaluating {n_pop} individuals...", flush=True)

        log_every = max(1, n_pop // 10)
        for idx, genotype in enumerate(pop):
            t_ind = time.time()
            fitnesses[idx] = world.evaluate_individual(
                genotype, n_repeats=n_repeats, n_steps=n_steps
            )
            scalar = float(fitnesses[idx].sum())
            if scalar > _best_scalar:
                _best_scalar = scalar
                shutil.copy2(join(world.temp_dir.name, "Robot.xml"), _best_xml_stage)

            if (idx + 1) % log_every == 0 or idx == n_pop - 1:
                f_str = "  ".join(f"{v:7.2f}" for v in fitnesses[idx])
                elapsed_ind = time.time() - t_ind
                print(
                    f"  [{idx+1:>{len(str(n_pop))}}/{n_pop}]"
                    f"  fit=[{f_str}]  sum={scalar:8.2f}"
                    f"  best_so_far={_best_scalar:8.2f}"
                    f"  ({elapsed_ind:.1f}s/ind)",
                    flush=True,
                )

        gen_elapsed = time.time() - t_gen_start
        gens_done = gen + 1
        gens_left = num_generations - gens_done
        avg_gen = (time.time() - t_run_start) / gens_done
        eta_s = avg_gen * gens_left
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))
        print(
            f"[Gen {gen+1}/{num_generations}]  done in {gen_elapsed:.1f}s"
            f"  |  best_sum={_best_scalar:.2f}"
            f"  |  ETA {eta_str}",
            flush=True,
        )

        save_ckpt = (gen % ckpt_interval == 0)
        ea.tell(pop, fitnesses, save_checkpoint=save_ckpt)

        if save_ckpt:
            ckpt_dir = join(results_dir, str(gen))
            # x_best.npy → 560 controller params only (TA-compatible)
            np.save(join(ckpt_dir, "x_best.npy"),
                    ea.x_best_so_far[:world.n_weights])
            # x_best_body.npy → 4 body params
            np.save(join(ckpt_dir, "x_best_body.npy"),
                    ea.x_best_so_far[world.n_weights:])
            shutil.copy2(_best_xml_stage, join(ckpt_dir, "Robot.xml"))

    # --- Training summary ---
    best_f     = ea.f_best_so_far
    score_path = join(results_dir, "training_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Final Project — Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generations     : {num_generations}\n")
        f.write(f"Population size : {population_size}\n")
        f.write(f"Controller      : {type(world.controller).__name__}"
                f"  ({world.n_weights} params)\n")
        f.write(f"Genotype size   : {world.n_params}"
                f"  (controller={world.n_weights}, body={world.n_body_params})\n\n")
        f.write("Best individual (highest sum of objectives):\n")
        labels = ["flat", "ice", "hill"]
        for label, val in zip(labels, best_f):
            f.write(f"  {label:<6}: {float(val):10.2f}\n")
        f.write(f"  {'sum':<6}: {float(best_f.sum()):10.2f}\n")
    print(f"\nTraining summary saved to: {score_path}")


if __name__ == "__main__":
    K = 4
    N_PER_GENOME = 32

    print(f"Loading top-{K} specialists per terrain from checkpoints...")
    flat_top_k, flat_scores = load_top_k_flat(K)
    ice_top_k,  ice_scores  = load_top_k_ice(K)
    hill_top_k, hill_scores = load_top_k_hill(K)

    print(f"  flat: {flat_top_k.shape}  scores={flat_scores.round(1)}")
    print(f"  ice : {ice_top_k.shape}   scores={ice_scores.round(1)}")
    print(f"  hill: {hill_top_k.shape}  scores={hill_scores.round(1)}")

    active = sum(1 for arr in [flat_top_k, ice_top_k, hill_top_k] if arr is not None)
    pop_size = 128 + active * K * N_PER_GENOME  # e.g. 128 + 3×4×32 = 512

    run_multi_task_evolution(
        num_generations=300,
        population_size=pop_size,
        n_parents=pop_size // 2,
        n_repeats=1,
        n_steps=500,
        ckpt_interval=10,
        bounds=(-10, 10),
        results_dir=join(ROOT_DIR, "results", "final_project"),
        random_seed=42,
        flat_top_k=flat_top_k,
        ice_top_k=ice_top_k,
        hill_top_k=hill_top_k,
        n_per_genome=N_PER_GENOME,
        ctrl_noise_std=0.05,
        body_noise_std=0.05,
    )

