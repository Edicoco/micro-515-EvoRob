"""
run_cluster.py  —  Production training entry point for JED HPC cluster.

Uses build_diverse_initial_population (4 best genomes per terrain, each
expanded to 32 individuals: 1 exact + 31 noisy) instead of the single-
specialist init in final_project_train.py.

If checkpoint files for a terrain are missing (results/ not rsynced),
that terrain's specialists are silently skipped and replaced by more
random individuals so training can still start.

Usage (local quick test):
    python run_cluster.py

Usage (cluster):
    sbatch run_final_project.sh
"""

import os
import shutil
import time
import xml.etree.ElementTree as xml
import numpy as np

from os.path import join

import evorob.world  # noqa: F401  — registers gym envs
from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import get_project_root

from final_project_train import FinalWorld, MAX_EPISODE_STEPS
from best_genome_selection import (
    load_top_k_flat,
    load_top_k_ice,
    load_top_k_hill,
    build_diverse_initial_population,
)

ROOT_DIR = get_project_root()

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------

NUM_GENERATIONS  = 1000
N_PER_GENOME     = 32      # 1 exact + 31 noisy  →  4 × 32 = 128 per terrain
K_SPECIALISTS    = 4       # top-k genomes per terrain
N_RANDOM         = 128
N_REPEATS        = 3       # parallel episodes per terrain per individual
N_STEPS          = 500
MUTATION_PROB    = 0.3
CROSSOVER_PROB   = 0.5
BOUNDS           = (-10, 10)
CKPT_INTERVAL    = 10
CTRL_NOISE_STD   = 0.1
BODY_NOISE_STD   = 0.1
RANDOM_SEED      = 42
RESULTS_DIR      = join(ROOT_DIR, "results", "final_project_cluster")

# ---------------------------------------------------------------------------
# Load top-k specialists (graceful fallback if checkpoints not present)
# ---------------------------------------------------------------------------

def _try_load(loader, name):
    try:
        genomes, scores = loader(K_SPECIALISTS)
        print(f"  {name}: top-{K_SPECIALISTS} scores = {scores.round(1)}")
        return genomes
    except Exception as exc:
        print(f"  {name}: skipped ({exc})")
        return None


print("Loading specialists...")
flat4 = _try_load(load_top_k_flat, "flat")
ice4  = _try_load(load_top_k_ice,  "ice")
hill4 = _try_load(load_top_k_hill, "hill")

# ---------------------------------------------------------------------------
# World + EA setup
# ---------------------------------------------------------------------------

np.random.seed(RANDOM_SEED)
world = FinalWorld()
print(f"\nGenotype: {world.n_params} params  (ctrl={world.n_weights}, body={world.n_body_params})")

n_active    = sum(t is not None for t in [flat4, ice4, hill4])
pop_size    = N_RANDOM + n_active * K_SPECIALISTS * N_PER_GENOME
n_parents   = pop_size // 4

print(f"Population: {N_RANDOM} random + {n_active} terrains × {K_SPECIALISTS} × {N_PER_GENOME} = {pop_size}")

ea = NSGAII(
    population_size=pop_size,
    n_opt_params=world.n_params,
    n_parents=n_parents,
    num_generations=NUM_GENERATIONS,
    bounds=BOUNDS,
    mutation_prob=MUTATION_PROB,
    crossover_prob=CROSSOVER_PROB,
    output_dir=RESULTS_DIR,
)

init_pop = build_diverse_initial_population(
    n_params=world.n_params,
    n_weights=world.n_weights,
    n_body_params=world.n_body_params,
    bounds=BOUNDS,
    n_random=N_RANDOM,
    flat_top_k=flat4,
    ice_top_k=ice4,
    hill_top_k=hill4,
    n_per_genome=N_PER_GENOME,
    ctrl_noise_std=CTRL_NOISE_STD,
    body_noise_std=BODY_NOISE_STD,
    random_seed=RANDOM_SEED,
)
ea.initialise_x0 = lambda: init_pop

# ---------------------------------------------------------------------------
# Evolution loop
# ---------------------------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)
_best_xml_stage = join(RESULTS_DIR, "_best_robot.xml")
_best_scalar    = -np.inf
n_obj           = 3

print(f"\nRunning {NUM_GENERATIONS} generations  pop={pop_size}  n_repeats={N_REPEATS}")
print(f"Objectives : [flat, ice, hill]")
print(f"Checkpoints: {RESULTS_DIR}\n")

t_run_start = time.time()

for gen in range(NUM_GENERATIONS):
    t_gen_start = time.time()
    pop   = ea.ask()
    n_pop = len(pop)
    fitnesses = np.empty((n_pop, n_obj))

    print(f"\n[Gen {gen+1}/{NUM_GENERATIONS}]  evaluating {n_pop} individuals...", flush=True)
    log_every = max(1, n_pop // 10)

    for idx, genotype in enumerate(pop):
        t_ind = time.time()
        fitnesses[idx] = world.evaluate_individual(genotype, n_repeats=N_REPEATS, n_steps=N_STEPS)
        scalar = float(fitnesses[idx].sum())

        if scalar > _best_scalar:
            _best_scalar = scalar
            shutil.copy2(join(world.temp_dir.name, "Robot.xml"), _best_xml_stage)

        if (idx + 1) % log_every == 0 or idx == n_pop - 1:
            f_str = "  ".join(f"{v:7.2f}" for v in fitnesses[idx])
            print(
                f"  [{idx+1:>{len(str(n_pop))}}/{n_pop}]"
                f"  fit=[{f_str}]  sum={scalar:8.2f}"
                f"  best_so_far={_best_scalar:8.2f}"
                f"  ({time.time()-t_ind:.1f}s/ind)",
                flush=True,
            )

    gen_elapsed = time.time() - t_gen_start
    gens_done   = gen + 1
    avg_gen     = (time.time() - t_run_start) / gens_done
    eta_str     = time.strftime("%H:%M:%S", time.gmtime(avg_gen * (NUM_GENERATIONS - gens_done)))
    print(
        f"[Gen {gen+1}/{NUM_GENERATIONS}]  done in {gen_elapsed:.1f}s"
        f"  |  best_sum={_best_scalar:.2f}"
        f"  |  ETA {eta_str}",
        flush=True,
    )

    save_ckpt = (gen % CKPT_INTERVAL == 0)
    ea.tell(pop, fitnesses, save_checkpoint=save_ckpt)

    if save_ckpt:
        ckpt_dir = join(RESULTS_DIR, str(gen))
        np.save(join(ckpt_dir, "x_best.npy"),      ea.x_best_so_far[:world.n_weights])
        np.save(join(ckpt_dir, "x_best_body.npy"), ea.x_best_so_far[world.n_weights:])
        shutil.copy2(_best_xml_stage, join(ckpt_dir, "Robot.xml"))

# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------

best_f     = ea.f_best_so_far
score_path = join(RESULTS_DIR, "training_score.txt")
with open(score_path, "w") as fh:
    fh.write("=" * 60 + "\n")
    fh.write("MICRO-515 Final Project — Cluster Training Summary\n")
    fh.write("=" * 60 + "\n\n")
    fh.write(f"Generations     : {NUM_GENERATIONS}\n")
    fh.write(f"Population size : {pop_size}\n")
    fh.write(f"n_repeats       : {N_REPEATS}\n")
    fh.write(f"K specialists   : {K_SPECIALISTS} per terrain\n")
    fh.write(f"n_per_genome    : {N_PER_GENOME}\n\n")
    fh.write("Best individual (highest sum of objectives):\n")
    for label, val in zip(["flat", "ice", "hill"], best_f):
        fh.write(f"  {label:<6}: {float(val):10.2f}\n")
    fh.write(f"  {'sum':<6}: {float(best_f.sum()):10.2f}\n")

print(f"\nTraining summary saved to: {score_path}")
