import numpy as np

class ABCOptimizer:
    def __init__(self, fitness_function, solution_dim, population_size=50, limit=10, max_iter=100, ranges=(-1, 1)):
        """
        Parameters:
        - fitness_function: Callable that takes a solution vector and returns a fitness value.
        - solution_dim: Dimensionality of the solution vector.
        - population_size: Number of food sources (solutions).
        - limit: Number of iterations after which a food source is abandoned.
        - max_iter: Maximum number of iterations.
        - ranges: Tuple indicating the range of values for the solution initialization.
        """
        self.fitness_function = fitness_function
        self.solution_dim = solution_dim
        self.population_size = population_size
        self.limit = limit
        self.max_iter = max_iter
        self.ranges = ranges
        self.population = np.random.uniform(ranges[0], ranges[1], (population_size, solution_dim))
        self.fitness = np.apply_along_axis(fitness_function, 1, self.population)
        self.trial_counters = np.zeros(population_size)
        self.best_solution = self.population[np.argmax(self.fitness)]
        self.best_fitness = np.max(self.fitness)

    def optimize(self):
        for iteration in range(self.max_iter):
            self._employed_bee_phase()
            fitness_probabilities = self._calculate_probabilities()
            self._onlooker_bee_phase(fitness_probabilities)
            self._scout_bee_phase()
            best_idx = np.argmax(self.fitness)
            if self.fitness[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_solution = self.population[best_idx].copy()

            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.best_fitness}")

        return self.best_solution, self.best_fitness

    def _employed_bee_phase(self):
        for i in range(self.population_size):
            self._explore(i)

    def _onlooker_bee_phase(self, fitness_probabilities):
        selected_indices = np.random.choice(
            range(self.population_size), size=self.population_size, p=fitness_probabilities
        )
        for i in selected_indices:
            self._explore(i)

    def _explore(self, i):
        k = np.random.choice([j for j in range(self.population_size) if j != i])
        phi = np.random.uniform(-1, 1, self.solution_dim)
        candidate = self.population[i] + phi * (self.population[i] - self.population[k])
        candidate = np.clip(candidate, self.ranges[0], self.ranges[1])
        candidate_fitness = self.fitness_function(candidate)

        if candidate_fitness > self.fitness[i]:
            self.population[i] = candidate
            self.fitness[i] = candidate_fitness
            self.trial_counters[i] = 0
        else:
            self.trial_counters[i] += 1

    def _calculate_probabilities(self, eps=1e-8):

        normalized_fitness = (self.fitness - self.fitness.min()) / (self.fitness.ptp() + eps)
        return normalized_fitness / (np.sum(normalized_fitness) + eps)

    def _scout_bee_phase(self):
        for i in range(self.population_size):
            if self.trial_counters[i] > self.limit:
                self.population[i] = np.random.uniform(self.ranges[0], self.ranges[1], self.solution_dim)
                self.fitness[i] = self.fitness_function(self.population[i])
                self.trial_counters[i] = 0
