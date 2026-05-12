"""
train_flat_specialist.py  —  CMA-ES specialist on flat terrain.

Objective: evolve a robust robot that advances STRAIGHT along the X axis.
Reward: see EvalFlatEnv.step (FlatEnv-v0).

Warm-start: loads best genome from --warm_start_dir if provided.
Output:     results/flat_specialist_cmaes/  (or --out_dir)

Usage (local test):
    python train_flat_specialist.py --n_gen 5 --pop_size 16 --n_steps 200

Usage (cluster):
    sbatch run_flat_specialist.sh
"""

import argparse
import copy
import os
import platform
import shutil
import time
import xml.etree.ElementTree as xml
from concurrent.futures import ProcessPoolExecutor
from os.path import join
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv

import evorob.world  # noqa: F401 — registers FlatEnv-v0
from evorob.algorithms.ea_api import CMAESAPI
from evorob.utils.filesys import get_project_root
from evorob.world.robot.controllers.mlp import NeuralNetworkController
from evorob.world.robot.morphology.ant_custom_robot import AntRobot
from final_project_train import FinalWorld, remap_challenge1_weights

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

ROOT_DIR    = get_project_root()
_ASSETS     = join(ROOT_DIR, "evorob", "world", "robot", "assets")

N_WEIGHTS     = 560
N_BODY_PARAMS = 0
N_PARAMS      = N_WEIGHTS

POP_SIZE      = 128
SIGMA0        = 0.1
BOUNDS        = (-10, 10)
N_GEN         = 2000
N_REPEATS     = 4
N_STEPS       = 1000
CKPT_INTERVAL = 10
RANDOM_SEED   = 42

SIGMA_RESTART  = 0.02   # restart CMA-ES when sigma drops below this

N_WORKERS = 0 if platform.system() == "Darwin" else max(1, (os.cpu_count() or 1) // (N_REPEATS + 1))

# ---------------------------------------------------------------------------
# Evaluation world (flat only)
# ---------------------------------------------------------------------------

class FlatSpecialistWorld(FinalWorld):
    """Evaluates a genotype on FlatEnv-v0 only."""

    def evaluate_individual(self, genotype: np.ndarray,
                            n_repeats: int = N_REPEATS,
                            n_steps: int = N_STEPS) -> float:
        self.update_robot_xml(genotype)
        return self._run_env("FlatEnv-v0", self.flat_world_file, n_repeats, n_steps)


# ---------------------------------------------------------------------------
# Per-worker state (needed for ProcessPoolExecutor with spawn)
# ---------------------------------------------------------------------------

_world_worker: FlatSpecialistWorld | None = None

def _init_worker():
    global _world_worker
    _world_worker = FlatSpecialistWorld()

def _eval_worker(args):
    genotype, n_repeats, n_steps = args
    fitness = _world_worker.evaluate_individual(genotype, n_repeats=n_repeats, n_steps=n_steps)
    with open(join(_world_worker.temp_dir.name, "Robot.xml")) as fh:
        robot_xml = fh.read()
    return fitness, robot_xml


# ---------------------------------------------------------------------------
# Warm-start helper
# ---------------------------------------------------------------------------

def _load_best_flat_body() -> np.ndarray:
    """Body params encoding the standard ant XML morphology (x_best_body.xml).

    Inverse of geno2pheno:  body_raw = (segment_param - 0.1) * 4 - 1
    where segment_param = xy_length * sqrt(2).

    Standard ant:  leg xy=0.20m  → param=0.2828 → raw=-0.269
                   ankle xy=0.40m → param=0.5657 → raw=+0.863
    """
    leg   = (0.20 * np.sqrt(2) - 0.1) * 4 - 1   # ≈ -0.269
    ankle = (0.40 * np.sqrt(2) - 0.1) * 4 - 1   # ≈ +0.863
    body = np.array([leg, ankle, leg, ankle])
    print(f"  body init center: standard ant XML  [leg={leg:.3f}, ankle={ankle:.3f}]")
    return body


def _load_warm_start(warm_start_dir: str | None) -> np.ndarray | None:
    if warm_start_dir is None:
        return None
    ctrl_path = join(warm_start_dir, "x_best.npy")
    if not os.path.isfile(ctrl_path):
        print(f"  warm_start: x_best.npy not found in {warm_start_dir}, starting random.")
        return None
    x0 = np.load(ctrl_path)[:N_WEIGHTS]
    print(f"  warm_start: remap from challenge 1 applied. If training is not based on a challenge 1 checkpoint, consider removing remap_challenge1_weights() to preserve original joint order.")
    print(f"  warm_start: loaded control params from {ctrl_path}  shape={x0.shape}")
    x0 = remap_challenge1_weights(x0)
    print(f"  warm_start: loaded from {warm_start_dir}  (Challenge-1 joint order remapped)  shape={x0.shape}")
    return x0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_gen: int, pop_size: int, n_repeats: int, n_steps: int,
         out_dir: str, warm_start_dir: str | None) -> None:

    np.random.seed(RANDOM_SEED)
    os.makedirs(out_dir, exist_ok=True)

    world = FlatSpecialistWorld()

    x0 = _load_warm_start(warm_start_dir)
    if x0 is None:
        x0 = np.random.uniform(BOUNDS[0], BOUNDS[1], N_PARAMS)

    es = CMAESAPI(
        n_params=N_PARAMS,
        population_size=pop_size,
        num_generations=n_gen,
        sigma=SIGMA0,
        bounds=BOUNDS,
        output_dir=out_dir,
        x0=x0,
    )
    es.es.opts.set({"seed": RANDOM_SEED, "verbose": -9, "tolx": 1e-6, "tolfun": 1e-6, "tolstagnation": 2000})

    mode_str = f"{N_WORKERS} workers" if N_WORKERS > 0 else "sequential (macOS)"
    print(f"\nCMA-ES flat specialist")
    print(f"  pop={pop_size}  sigma0={SIGMA0}  n_gen={n_gen}")
    print(f"  n_repeats={n_repeats}  n_steps={n_steps}")
    print(f"  eval mode: {mode_str}  (cores={os.cpu_count()})")
    print(f"  output: {out_dir}\n")

    best_fitness = -np.inf
    best_xml     = None
    best_genome  = None
    n_restarts   = 0
    _best_xml_stage = join(out_dir, "_best_robot.xml")

    pool_ctx = (
        ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker)
        if N_WORKERS > 0 else None
    )

    t_start = time.time()

    try:
        for gen in range(n_gen):
            if es.es.stop():
                print("CMA-ES stop condition reached.")
                break

            solutions = es.ask()  # (pop, N_PARAMS) — bounds handled by CMA-ES
            t_gen = time.time()

            print(f"[Gen {gen+1}/{n_gen}]  evaluating {len(solutions)} individuals  [{mode_str}]...", flush=True)

            if pool_ctx is not None:
                args = [(g, n_repeats, n_steps) for g in solutions]
                results = list(pool_ctx.map(_eval_worker, args, chunksize=1))
                fitnesses  = np.array([r[0] for r in results])
                xml_strs   = [r[1] for r in results]
            else:
                fitnesses = np.empty(len(solutions))
                xml_strs  = []
                log_every = max(1, len(solutions) // 8)
                for idx, g in enumerate(solutions):
                    t0 = time.time()
                    f  = world.evaluate_individual(g, n_repeats=n_repeats, n_steps=n_steps)
                    with open(join(world.temp_dir.name, "Robot.xml")) as fh:
                        xml_strs.append(fh.read())
                    fitnesses[idx] = f
                    if (idx + 1) % log_every == 0 or idx == len(solutions) - 1:
                        print(f"  [{idx+1}/{len(solutions)}]  fit={f:8.2f}  ({time.time()-t0:.1f}s/ind)", flush=True)

            es.tell(solutions, fitnesses)  # CMAESAPI negates internally

            best_idx = int(fitnesses.argmax())
            gen_best = float(fitnesses[best_idx])
            if gen_best > best_fitness * 0.95:
                # Re-evaluate candidate with 2× repeats to filter lucky outliers.
                candidate = solutions[best_idx]
                if pool_ctx is not None:
                    confirmed = list(pool_ctx.map(
                        _eval_worker, [(candidate, n_repeats * 2, n_steps)], chunksize=1
                    ))[0][0]
                else:
                    confirmed = world.evaluate_individual(candidate, n_repeats=n_repeats * 2, n_steps=n_steps)
                if confirmed > best_fitness:
                    best_fitness = confirmed
                    best_genome  = candidate.copy()
                    best_xml     = xml_strs[best_idx]
                    with open(_best_xml_stage, "w") as fh:
                        fh.write(best_xml)
                gen_best = confirmed

            elapsed = time.time() - t_gen
            avg_gen = (time.time() - t_start) / (gen + 1)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(avg_gen * (n_gen - gen - 1)))
            print(
                f"[Gen {gen+1}/{n_gen}]  done in {elapsed:.1f}s"
                f"  |  gen_best={gen_best:.2f}"
                f"  |  best_so_far={best_fitness:.2f}"
                f"  |  sigma={es.es.sigma:.3f}"
                f"  |  ETA {eta_str}",
                flush=True,
            )

            if gen % CKPT_INTERVAL == 0 and best_genome is not None:
                ckpt = join(out_dir, str(gen))
                os.makedirs(ckpt, exist_ok=True)
                np.save(join(ckpt, "x_best.npy"), best_genome)
                np.save(join(ckpt, "x.npy"),      solutions)
                np.save(join(ckpt, "f.npy"),      fitnesses)
                if best_xml:
                    with open(join(ckpt, "Robot.xml"), "w") as fh:
                        fh.write(best_xml)

            if es.es.sigma < SIGMA_RESTART and best_genome is not None:
                print(f"[Gen {gen+1}] sigma={es.es.sigma:.4f} < {SIGMA_RESTART} → convergé, arrêt.")
                break

    finally:
        if pool_ctx is not None:
            pool_ctx.shutdown(wait=False)

    # Final save
    if best_genome is not None:
        final_dir = join(out_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        np.save(join(final_dir, "x_best.npy"), best_genome)
        if best_xml:
            with open(join(final_dir, "Robot.xml"), "w") as fh:
                fh.write(best_xml)
        print(f"\nBest fitness: {best_fitness:.2f}")
        print(f"Final checkpoint: {final_dir}")
        print(f"Test with: python final_project_test.py --best_dir_path {final_dir}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gen",          type=int,   default=N_GEN)
    parser.add_argument("--pop_size",       type=int,   default=POP_SIZE)
    parser.add_argument("--n_repeats",      type=int,   default=N_REPEATS)
    parser.add_argument("--n_steps",        type=int,   default=N_STEPS)
    parser.add_argument("--out_dir",        type=str,   default=join(ROOT_DIR, "results", "flat_specialist_cmaes"))
    parser.add_argument("--warm_start_dir", type=str,   default=join(ROOT_DIR, "results", "best_flat", "2999"),
                        help="Directory with x_best.npy to warm-start CMA-ES")
    args = parser.parse_args()
    main(
        n_gen          = args.n_gen,
        pop_size       = args.pop_size,
        n_repeats      = args.n_repeats,
        n_steps        = args.n_steps,
        out_dir        = args.out_dir,
        warm_start_dir = args.warm_start_dir,
    )
