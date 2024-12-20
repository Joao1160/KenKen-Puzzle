import random
import numpy as np

# Define KenKen cage constraints (adjusted for a 5x5 grid)
cages = [
    ('*', 12, [(0, 0), (1, 0)]),
    ('-', 2, [(0, 1), (0, 2)]),
    ('-', 1, [(0, 3), (1, 3)]),
    ('/', 2, [(1, 1), (1, 2)]),
    ('-', 1, [(2, 0), (2, 1)]), 
    ('/', 2, [(2, 2), (3, 2)]), 
    ('*', 12, [(2, 3), (3, 3)]), 
    ('+', 5, [(3, 0), (3, 1)]),
]
# Generate an individual grid using numpy for better performance
def generate_individual(grid_size, seed=None):
    if seed is not None:
        random.seed(seed)

    while True:
        grid = np.zeros((grid_size, grid_size), dtype=int)
        for i in range(grid_size):
            grid[i] = np.random.permutation(grid_size) + 1  # Generate a random permutation of numbers 1 to grid_size
        
        # Ensure rows and columns are unique using numpy functions
        if np.all(np.unique(grid, axis=0).shape[0] == grid_size) and np.all(np.unique(grid, axis=1).shape[0] == grid_size):
            break
    return grid

# Vectorized uniqueness check using NumPy
def check_uniqueness(solution, grid_size):
    fitness_penalty = 0
    grid = solution
    
    for i in range(grid_size):
        row_values = set()
        col_values = set()
        for j in range(grid_size):
            value_row = grid[i, j]
            
            if value_row in row_values:
                fitness_penalty += 1
            row_values.add(value_row)

            value_col = grid[j, i]

            if value_col in col_values:
                fitness_penalty += 1
            col_values.add(value_col)

    return fitness_penalty


# Calculate cage result based on operation
def calculate_cage_result(cage_values, operation):
    if operation == "=":
        return sum(cage_values)
    elif operation == "+":
        return sum(cage_values)
    elif operation == "-":
        return abs(cage_values[0] - cage_values[1]) if len(cage_values) == 2 else 0
    elif operation == "*":
        result = 1
        for value in cage_values:
            result *= value
        return result
    elif operation == "/":
        if len(cage_values) == 2:
            larger, smaller = max(cage_values), min(cage_values)
            return larger // smaller if smaller != 0 and larger % smaller == 0 else 0
        return 0
    else:
        raise ValueError(f"Unsupported operation: {operation}")


# Vectorized evaluation of cage constraints using numpy
#    ('*', 12, [(2, 3), (3, 3)])
def evaluate_cage(solution, cages):
    penalties = 0
    for operation, target, cells in cages:
        cage_values = np.array([solution[i][j] for i, j in cells])
        if calculate_cage_result(cage_values, operation) != target:
            penalties += 1
    return penalties

# Evaluate fitness of a solution
def evaluate_fitness(solution, grid_size, cages):
    uniqueness_penalty = check_uniqueness(solution, grid_size)
    cage_penalty = evaluate_cage(solution, cages)
    return uniqueness_penalty + cage_penalty

# Tournament selection function
def selection(population, fitness_values):
    # Pair individuals with their fitness values
    paired_population = list(zip(population, fitness_values))
    
    # Sort by fitness (ascending order: lower fitness is better)
    sorted_population = sorted(paired_population, key=lambda x: x[1])
    
    # Select the top 50%
    half_population_size = len(population) // 2
    selected_population = [individual for individual, _ in sorted_population[:half_population_size]]
    
    return selected_population


# Crossover between two parents
def crossover(parent1, parent2):
    split = random.randint(1, len(parent1) - 1) //3
    child1 = np.vstack((parent1[:split], parent2[split:]))
    child2 = np.vstack((parent2[:split], parent1[split:]))
    return child1, child2

# Mutation function
def mutate(solution, pm):
    grid_size = len(solution)
    mutated_solution = solution.copy()
    for i in range(grid_size):
        if random.random() < pm:
            idx1, idx2 = random.sample(range(grid_size), 2)
            mutated_solution[i][idx1], mutated_solution[i][idx2] = (mutated_solution[i][idx2],mutated_solution[i][idx1],
            )
    return mutated_solution 

# Main genetic algorithm function
def kenken(pop_size, max_generations, grid_size, cages, pm):
    population = [generate_individual(grid_size) for _ in range(pop_size)]
    best_fitness_overall = float("inf") # 3
    best_solution = None

    for generation in range(max_generations):
        fitness_values = [evaluate_fitness(ind, grid_size, cages) for ind in population]

        # Update best solution
        min_fitness = min(fitness_values) # 3
        if min_fitness < best_fitness_overall:
            best_fitness_overall = min_fitness
            best_solution = population[fitness_values.index(min_fitness)]

        print(f"Generation {generation:03}: Best fitness = {best_fitness_overall}")

        # Terminate early if optimal solution is found
        if best_fitness_overall == 0:
            print("Optimal solution found!")
            return best_solution

        # Selection
        selected_population = selection(population, fitness_values)

        # Crossover and Mutation
        children = []
        while len(children) < pop_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2)
            children.append(mutate(child1, pm))
            children.append(mutate(child2, pm))

        population = children[:pop_size]

    print("Final population reached. Best solution:")
    return best_solution

# Main function to run the program
def main():
    pop_size = 100  # Increased population size for larger grids
    max_generations = 1000  # Reduced number of generations
    grid_size = 4 # 5x5 KenKen grid
    mutation_prob = 0.1

    print("Starting KenKen genetic algorithm...")
    solution = kenken(pop_size, max_generations, grid_size, cages, mutation_prob)

    print("\nFinal Solution:")
    print(solution)

    # Verify solution correctness
    if check_uniqueness(solution, grid_size) == 0 and evaluate_cage(solution, cages) == 0:
        print("Solution is 100% correct!")
    else:
        print("Solution failed verification. Please review the algorithm.")

if __name__ == "__main__":
    main()


