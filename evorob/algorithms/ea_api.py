import numpy as np
import cma
from evorob.algorithms.base_ea import EA


class EvoAlgAPI(EA):
    """Evolutionary algorithm API wrapper.

    This class provides an interface to wrap any EA framework that uses
    the ask-tell pattern (CMA-ES, pyribs, evosax, etc.).

    Example frameworks to use:
    - CMA-ES: https://github.com/CMA-ES/pycma
    - pyribs: https://github.com/icaros-usc/pyribs/
    - evosax: https://github.com/RobertTLange/evosax/
    - EvoJAX: https://github.com/google/evojax
    """

    def __init__(self, n_params: int, population_size: int = 100, num_generations: int = 100,
                 output_dir: str = "./results/EA", **kwargs):
        """Initialize the evolutionary algorithm.

        Args:
            n_params: Dimensionality of the search space
            population_size: Number of solutions per generation
            num_generations: Number of generations
            output_dir: Directory for saving checkpoints
            **kwargs: Additional arguments for the EA framework
        """
        # TODO: Initialize your chosen EA framework here
        self.n_params = n_params
        self.n_gen = num_generations
        self.population_size = population_size
        
        # % bookkeeping for base EA
        self.directory_name = output_dir
        self.current_gen = 0
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

        # Initialisation de CMA-ES
        # x0 = np.random.uniform(-0.2, 0.2, n_params)  
        x0 = np.load("results/20260315_211559_neural_controller_ckpts/2999/x_best.npy") # Meilleure initialisation
        sigma0 = kwargs.get('sigma0', 0.15)  # Valeur par défaut si non fournie

        """
        self.es = cma.CMAEvolutionStrategy(
            x0,
            sigma0,
            {
                'popsize': population_size,  # Utilisez population_size passé à EvoAlgAPI
                'CMA_diagonal': kwargs.get('CMA_diagonal', False),  # Désactivé par défaut
                'tolx': kwargs.get('tolx', 1e-6),
                'tolfun': kwargs.get('tolfun', 1e-6),
                'verbose': -9,
            }
        )
        """

        inopts={'popsize': population_size}#, 'seed': 42}
        self.es = cma.CMAEvolutionStrategy(x0,sigma0,inopts)

        # % 

    def ask(self) -> np.ndarray:
        """Sample population from the algorithm.

        Returns:
            population: Array of shape (population_size, n_params)
                       Each row is a candidate solution
        """
        # TODO: Get new population from your EA
        # Make sure the returned array has shape (population_size, n_params)

        self.current_population = self.es.ask()
        return np.array(self.current_population)

    def tell(self, population: np.ndarray, fitnesses: np.ndarray, save_checkpoint: bool = False) -> None:
        """Update the algorithm with evaluated population.

        Args:
            population: Array of shape (population_size, n_params)
            fitnesses: Array of shape (population_size,) with fitness values
                      Higher is better (maximization)
            save_checkpoint: Whether to save checkpoint after update
        """
        # TODO: Update your EA with the evaluated population
        # Note: Some algorithms minimize, others maximize.
        # Adjust accordingly (negate fitnesses if needed).
        self.es.tell(self.current_population, -fitnesses)

        # After updating the EA, do bookkeeping for checkpointing:
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

        return

