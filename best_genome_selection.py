"""
best_genome_selection.py
========================
Utilities to extract the top-k genomes per terrain from training checkpoints
and build a diverse initial population for NSGA-II multi-task evolution.

Population structure (per terrain, k=4, n_per_genome=32):
    k exact copies  (controller + neutral body)   →  k          individuals
    k × (n_per_genome - 1) noisy variants         →  k × 31     individuals
    ─────────────────────────────────────────────────────────────────────────
    Total per terrain                              →  k × 32 = 128 individuals

Full population with 3 terrains:
    128 random  +  128 flat  +  128 ice  +  128 hill  =  512
"""

import os
import numpy as np
from os.path import join

from evorob.utils.filesys import get_project_root

ROOT_DIR  = get_project_root()
_RESULTS  = join(ROOT_DIR, "results")

FLAT_CKPT_DIR = join(_RESULTS, "best_flat")
HILL_CKPT_DIR = join(_RESULTS, "best_hill")
ICE_CKPT_DIR  = join(_RESULTS, "20260324_085356_nsga_ckpts")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_gen_dir(ckpt_dir: str) -> str:
    """Return the path of the highest-numbered generation sub-directory."""
    gens = sorted([int(d) for d in os.listdir(ckpt_dir) if d.isdigit()])
    if not gens:
        raise FileNotFoundError(f"No numbered generation dirs found in {ckpt_dir}")
    return join(ckpt_dir, str(gens[-1]))


# ---------------------------------------------------------------------------
# Top-k extraction
# ---------------------------------------------------------------------------

def get_top_k(ckpt_dir: str, k: int = 4, obj_col: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the top-k genomes and their scores from the last checkpoint.

    Args:
        ckpt_dir:  Path containing numbered generation sub-directories with
                   x.npy (pop, n_weights) and f.npy (pop,) or (pop, n_obj).
        k:         Number of genomes to return.
        obj_col:   Column of f.npy to use as scalar fitness when f is 2-D.
                   None → f.npy is 1-D.

    Returns:
        genomes : np.ndarray  shape (k, n_weights)
        scores  : np.ndarray  shape (k,)
    """
    last = _last_gen_dir(ckpt_dir)
    x = np.load(join(last, "x.npy"))   # (pop, n_weights)
    f = np.load(join(last, "f.npy"))   # (pop,) or (pop, n_obj)

    scores = f[:, obj_col] if (f.ndim == 2 and obj_col is not None) else f
    top_idx = np.argsort(scores)[-k:][::-1]  # descending — best first
    return x[top_idx], scores[top_idx]


def load_top_k_flat(k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Top-k flat specialists (single-objective CMA-ES run)."""
    return get_top_k(FLAT_CKPT_DIR, k=k)


def load_top_k_hill(k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Top-k hill specialists (single-objective CMA-ES run)."""
    return get_top_k(HILL_CKPT_DIR, k=k)


def load_top_k_ice(k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Top-k ice specialists from the 2-objective (flat, ice) NSGA-II run.

    Ranks by ice objective (column 1 of f.npy).
    """
    return get_top_k(ICE_CKPT_DIR, k=k, obj_col=1)


# ---------------------------------------------------------------------------
# Diverse population builder
# ---------------------------------------------------------------------------

def build_diverse_initial_population(
    n_params: int,
    n_weights: int,
    n_body_params: int,
    bounds: tuple,
    n_random: int = 128,
    flat_top_k: np.ndarray | None = None,
    ice_top_k: np.ndarray | None = None,
    hill_top_k: np.ndarray | None = None,
    n_per_genome: int = 32,
    ctrl_noise_std: float = 0.1,
    body_noise_std: float = 0.1,
    random_seed: int = 42,
) -> np.ndarray:
    """Build initial population with k diverse specialists per terrain.

    For each terrain with k provided genomes:
        1 exact copy per genome   (controller + neutral body)
        (n_per_genome - 1) noisy copies per genome

    Args:
        n_params:       Full genotype length (ctrl + body).
        n_weights:      Controller-only length.
        n_body_params:  Body-only length.
        bounds:         (min, max) clipping range applied to every individual.
        n_random:       Fully random individuals at the front of the population.
        flat_top_k:     (k, n_weights) flat specialists, or None to skip.
        ice_top_k:      (k, n_weights) ice  specialists, or None to skip.
        hill_top_k:     (k, n_weights) hill specialists, or None to skip.
        n_per_genome:   Individuals derived from one genome  (1 exact + rest noisy).
        ctrl_noise_std: σ of Gaussian noise added to controller weights.
        body_noise_std: σ of Gaussian noise added to body params.
        random_seed:    RNG seed for reproducibility.

    Returns:
        np.ndarray of shape  (n_random + n_active_terrains × k × n_per_genome, n_params).
        With defaults (k=4, n_per_genome=32, 3 terrains):  128 + 3×128 = 512.
    """
    rng = np.random.default_rng(random_seed)
    parts = []

    # --- Fully random individuals ---
    rand = rng.uniform(bounds[0], bounds[1], (n_random, n_params))
    rand[:, n_weights:] = rng.normal(0, body_noise_std, (n_random, n_body_params))
    parts.append(np.clip(rand, bounds[0], bounds[1]))

    # --- Terrain specialists ---
    neutral_body = np.zeros(n_body_params)
    terrain_labels = ["flat", "ice", "hill"]
    terrain_arrays = [flat_top_k, ice_top_k, hill_top_k]

    for label, terrain_genomes in zip(terrain_labels, terrain_arrays):
        if terrain_genomes is None:
            continue
        k = len(terrain_genomes)
        terrain_parts = []
        for rank, ctrl in enumerate(terrain_genomes):
            # 1 exact copy with neutral body
            exact = np.clip(np.concatenate([ctrl, neutral_body]), bounds[0], bounds[1])
            genome_group = [exact]

            # (n_per_genome - 1) noisy variants
            for _ in range(n_per_genome - 1):
                noisy_ctrl = ctrl + rng.normal(0, ctrl_noise_std, n_weights)
                noisy_body = rng.normal(0, body_noise_std, n_body_params)
                noisy = np.clip(np.concatenate([noisy_ctrl, noisy_body]), bounds[0], bounds[1])
                genome_group.append(noisy)

            terrain_parts.append(np.array(genome_group))  # (n_per_genome, n_params)

        terrain_block = np.vstack(terrain_parts)  # (k × n_per_genome, n_params)
        print(f"  {label:4s}: {k} genomes × {n_per_genome} = {len(terrain_block)} individuals")
        parts.append(terrain_block)

    population = np.vstack(parts)
    return population


# ---------------------------------------------------------------------------
# Standalone demo / verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    K = 4

    print(f"Loading top-{K} specialists per terrain from last checkpoint...")

    flat4, flat_scores = load_top_k_flat(K)
    ice4,  ice_scores  = load_top_k_ice(K)
    hill4, hill_scores = load_top_k_hill(K)

    print(f"\n  flat : {flat4.shape}  scores = {flat_scores.round(1)}")
    print(f"  ice  : {ice4.shape}   scores = {ice_scores.round(1)}")
    print(f"  hill : {hill4.shape}  scores = {hill_scores.round(1)}")

    n_weights    = 560
    n_body_params = 4
    n_params     = n_weights + n_body_params   # 564

    pop = build_diverse_initial_population(
        n_params=n_params,
        n_weights=n_weights,
        n_body_params=n_body_params,
        bounds=(-10, 10),
        n_random=128,
        flat_top_k=flat4,
        ice_top_k=ice4,
        hill_top_k=hill4,
        n_per_genome=32,       # 1 exact + 31 noisy per genome
        ctrl_noise_std=0.1,
        body_noise_std=0.1,
        random_seed=42,
    )

    print(f"\nTotal population shape : {pop.shape}")
    # Expected: 128 random + 3 × (4 × 32) = 128 + 384 = 512
    print(f"Expected               : {128 + 3 * K * 32}")
    assert pop.shape == (128 + 3 * K * 32, n_params), "Population size mismatch!"
    print("OK — population looks correct.")

