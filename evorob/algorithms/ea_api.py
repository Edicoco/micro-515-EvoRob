from curses import raw

import numpy as np
import cma
from evorob.algorithms.base_ea import EA
from evorob.utils.filesys import search_file_list
import os

class CMAESAPI(EA):
    """Wrapper for the original CMA-ES library (cma package)."""

    def __init__(
        self,
        n_params: int,
        population_size: int,
        num_generations: int = 100,
        sigma: float = 0.15,
        bounds: tuple[int, int] = (-10, 10),
        output_dir: str = "./results/CMAES",
    ):
        self.population_size = population_size
        self.n_gen = num_generations
        self.n_params = n_params

        # % bookkeeping for base EA
        self.directory_name = output_dir
        self.current_gen = 0
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

        # Initialize with random mean
        # initial_mean = np.random.uniform(bounds[0], bounds[1], n_params)
        initial_mean = np.load("x_best.npy")
        initial_mean = np.clip(initial_mean, bounds[0], bounds[1])

        # Create CMA-ES optimizer
        opts = {"popsize": population_size, "bounds": bounds}
        self.es = cma.CMAEvolutionStrategy(x0=initial_mean, sigma0=sigma, inopts=opts)

    def ask(self):
        """Sample population from CMA-ES."""
        population = self.es.ask()
        return np.array(population)

    def tell(self, population, fitnesses, save_checkpoint: bool = False):
        """Update CMA-ES with evaluated population
        Note: CMA-ES minimizes, so negate fitnesses."""
        self.es.tell(population.tolist(), (-fitnesses).tolist())

        # % bookkeeping for checkpointing
        self.full_f.append(fitnesses)
        self.full_x.append(population)
        self.f = fitnesses
        self.x = population

        # Track best individual
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.f_best_so_far:
            self.f_best_so_far = fitnesses[best_idx]
            self.x_best_so_far = population[best_idx].copy()

        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1

    def load_checkpoint(self):
        dir_path = search_file_list(self.directory_name, 'f_best.npy')
        assert len(dir_path) > 0;
        "No files are here, check the directory_name!!"

        self.current_gen = int(dir_path[-1].split('/')[-2])
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        print(f"Loading from: {curr_gen_path}")
        self.full_f = np.load(os.path.join(self.directory_name, 'full_f.npy'))
        self.full_x = np.load(os.path.join(self.directory_name, 'full_x.npy'))
        self.f_best_so_far = np.load(os.path.join(curr_gen_path, 'f_best.npy'))
        self.x_best_so_far = np.load(os.path.join(curr_gen_path, 'x_best.npy'))
        self.x = np.load(os.path.join(curr_gen_path, 'x.npy'))
        self.f = np.load(os.path.join(curr_gen_path, 'f.npy'))

        self.cmaes = self.load_cmeas()
        for x, f in zip(self.full_x, self.full_f):
            self.cmaes.tell(x, f)

EvoAlgAPI = CMAESAPI

