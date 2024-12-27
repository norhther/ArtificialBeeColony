import torch
import torch.nn as nn
from ABCOptimizer import ABCOptimizer

def fitness_function(solution):
    model = nn.Linear(10, 1)
    model.weight.data = torch.tensor(solution[:10], dtype=torch.float32).reshape(1, -1)
    model.bias.data = torch.tensor(solution[10:], dtype=torch.float32)

    inputs = torch.randn(100, 10)
    targets = torch.randn(100, 1)
    criterion = nn.MSELoss()
    outputs = model(inputs)
    loss = -criterion(outputs, targets).item() 
    return loss

solution_dim = 11  # 10 weights + 1 bias
abc = ABCOptimizer(fitness_function, solution_dim, population_size=20, max_iter=50)
best_solution, best_fitness = abc.optimize()

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
