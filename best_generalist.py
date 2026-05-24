"""
best_generalist.py
==================
Select and save the best generalist genomes from a multi-objective
(flat, ice, hill) NSGA-II checkpoint directory.

Five selection strategies:
  1. maximin      — maximise the worst objective  (max min_i f_i)
  2. mean2worst   — maximise the mean of the two worst objectives
  3. best_flat    — best on flat terrain   (obj 0)
  4. best_ice     — best on ice terrain    (obj 1)
  5. best_hill    — best on hill terrain   (obj 2)

Each selected genome is saved as a sub-directory compatible with
final_project_test.py (contains x_best.npy, x_best_body.npy, Robot.xml).

Usage:
    python best_generalist.py
    python best_generalist.py --source results/final_project_cluster
    python best_generalist.py --source results/final_project_cluster --out results/generalists
"""

import argparse
import os
import shutil
import numpy as np
from os.path import join

# Controller weights / body params split (must match training config)
N_WEIGHTS     = 560
N_BODY_PARAMS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_gen_dir(ckpt_dir: str) -> str:
    gens = sorted([int(d) for d in os.listdir(ckpt_dir) if d.isdigit()])
    if not gens:
        raise FileNotFoundError(f"No numbered generation dirs in {ckpt_dir}")
    return join(ckpt_dir, str(gens[-1])), gens[-1]


def _score_maximin(f: np.ndarray) -> np.ndarray:
    return f.min(axis=1)


def _score_mean2worst(f: np.ndarray) -> np.ndarray:
    return np.sort(f, axis=1)[:, :2].mean(axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(source_dir: str, out_dir: str) -> None:
    last_dir, last_gen = _last_gen_dir(source_dir)
    print(f"Source : {source_dir}")
    print(f"Using last checkpoint : generation {last_gen}  ({last_dir})")

    x = np.load(join(last_dir, "x.npy"))   # (pop, n_params)
    f = np.load(join(last_dir, "f.npy"))   # (pop, 3)
    robot_xml_src = join(last_dir, "Robot.xml")
    has_xml = os.path.isfile(robot_xml_src)

    print(f"Population : {x.shape[0]} individuals  |  objectives : [flat, ice, hill]")
    print(f"Score ranges : flat [{f[:,0].min():.1f}, {f[:,0].max():.1f}]"
          f"  ice [{f[:,1].min():.1f}, {f[:,1].max():.1f}]"
          f"  hill [{f[:,2].min():.1f}, {f[:,2].max():.1f}]\n")

    strategies = [
        ("maximin",    _score_maximin(f)),
        ("mean2worst", _score_mean2worst(f)),
        ("best_flat",  f[:, 0]),
        ("best_ice",   f[:, 1]),
        ("best_hill",  f[:, 2]),
    ]

    os.makedirs(out_dir, exist_ok=True)

    header = f"{'Strategy':<14}  {'flat':>8}  {'ice':>8}  {'hill':>8}  {'min':>8}  {'mean2w':>8}"
    print(header)
    print("-" * len(header))

    for name, scores in strategies:
        idx = int(np.argmax(scores))
        genome = x[idx]                          # (n_params,)
        ctrl   = genome[:N_WEIGHTS]              # (n_weights,)
        body   = genome[N_WEIGHTS:]              # (n_body_params,)
        fi     = f[idx]

        dest = join(out_dir, name)
        os.makedirs(dest, exist_ok=True)
        np.save(join(dest, "x_best.npy"),      ctrl)
        np.save(join(dest, "x_best_body.npy"), body)
        if has_xml:
            shutil.copy2(robot_xml_src, join(dest, "Robot.xml"))

        min2w = float(np.sort(fi)[:2].mean())
        print(f"{name:<14}  {fi[0]:>8.1f}  {fi[1]:>8.1f}  {fi[2]:>8.1f}"
              f"  {fi.min():>8.1f}  {min2w:>8.1f}")

    print(f"\nSaved to : {out_dir}")
    print("Each sub-directory is compatible with:")
    print("  python final_project_test.py --best_dir_path <sub-dir>")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from evorob.utils.filesys import get_project_root
    ROOT = get_project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default=join(ROOT, "results/NSGA/run_02"),
        help="Directory containing numbered generation checkpoints",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: same as --source)",
    )
    args = parser.parse_args()
    out = args.out if args.out else args.source

    main(args.source, out)
