function generateBoard(cagesObject) {
  const cages = JSON.parse(cagesObject)

  localStorage.setItem('Cages', JSON.stringify(cagesObject));

  const container = document.getElementById("container");
  const size = parseInt(document.getElementById("sizeSelect").value);

  container.innerHTML = "";

  // Set grid template
  container.style.gridTemplateColumns = `repeat(${size}, 1fr)`;
  container.style.gridTemplateRows = `repeat(${size}, 1fr)`;

  let arrayOfCagesID = [];
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      for (let x = 0; x < cages.length; x++) {
        if (Array.isArray(cages[x][2])) {
          for (let y = 0; y < cages[x][2].length ; y++) {
            if (i === cages[x][2][y][0] && j === cages[x][2][y][1]) {
              const square = document.createElement("div");
              square.className = "grid-square";

              const hue = Math.floor((x * 360) / cages.length); // Spread `x` evenly across 360 degrees
              square.style.backgroundColor = `hsl(${hue}, 70%, 80%)`;
              if (arrayOfCagesID.includes(x)) {
                square.innerHTML = "";
              } else {
                if (cages[x][0] == "/") {
                  cages[x][0] = "÷";
                } else if (cages[x][0] == "*") {
                  cages[x][0] = "x";
                }
                square.innerHTML = `<span>${cages[x][1]}${cages[x][0]}</span>`;
              }
              container.appendChild(square);
              arrayOfCagesID.push(x);

              break;
            }
          }
        }
      }
    }
  }
}

async function solve() {
    cages = localStorage.getItem('Cages');

    const pyodide = await loadPyodide();
    const pythonCode = `
import random
import numpy as np
import json


# Define KenKen cage constraints (adjusted for a 5x5 grid)
cages = json.loads(${cages})
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
    split = random.randint(1, len(parent1) - 1)
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
            mutated_solution[i][idx1], mutated_solution[i][idx2] = (
                mutated_solution[i][idx2],
                mutated_solution[i][idx1],
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


pop_size = ${parseInt(document.getElementById("popSize").value)}  # Increased population size for larger grids 
max_generations = ${parseInt(document.getElementById("maxGeneration").value)}  # Reduced number of generations
grid_size = ${parseInt(document.getElementById("sizeSelect").value)} # 5x5 KenKen grid
mutation_prob = 0.1

solution = kenken(pop_size, max_generations, grid_size, cages, mutation_prob)

newCages = json.dumps(solution.tolist())  # Convert ndarray to list

newCages


    `;
    
    await pyodide.loadPackage("numpy")

    const result = await pyodide.runPythonAsync(pythonCode);

    console.log(result);

    generateSolvedBoard(result)
    }
function generateSolvedBoard(cagesObject) {
  const board = JSON.parse(cagesObject);  
  const board1D = board.reduce((flatArray, row) => flatArray.concat(row), [])
    const gridSquares = document.querySelectorAll('.grid-square');
    gridSquares.forEach((square, index) => {
        square.innerHTML = square.innerHTML + board1D[index]
    });

}

function hide() {
  const container = document.getElementById("container");
  container.innerHTML = ""; // Clears the grid
}

async function runPython() {
  const pyodide = await loadPyodide();
  const pythonCode = `
import random
import json


# Generate random horizontal cages
def generate_horizontal_cages(n):
    cages = []
    for row in range(n):
        for col in range(n - 1):
            cage = [(row, col), (row, col + 1)]
            cages.append(cage)
    return cages

# Generate random vertical cages
def generate_vertical_cages(n):
    cages = []
    for col in range(n):
        for row in range(n - 1):
            cage = [(row, col), (row + 1, col)]
            cages.append(cage)
    return cages

# Choose a random operator from +, -, *, /
def random_operator():
    return random.choice(['+', '-', '*', '/'])

# Combine horizontal and vertical cages while ensuring no overlap of cells
def generate_random_cages_with_operators(n):
    cages = []
    visited = set()

    # Generate all potential cages
    horizontal_cages = generate_horizontal_cages(n)
    vertical_cages = generate_vertical_cages(n)

    # Combine horizontal and vertical cages into a single pool
    all_cages = horizontal_cages + vertical_cages
    random.shuffle(all_cages)  # Shuffle for randomness

    # Randomly select cages while avoiding overlaps
    for cage in all_cages:
        if not any(cell in visited for cell in cage):
            operator = random_operator()  # Assign a random operator
            cages.append((cage, operator))
            visited.update(cage)

    # Add single-cell cages for any uncovered positions
    for row in range(n):
        for col in range(n):
            if (row, col) not in visited:
                # Add this position as a single-cell cage with no operator and its number as the result
                cages.append([[(row, col)], None])  # No operator and result is the number
                visited.add((row, col))

    return cages

# Apply an operator to a list of numbers
def apply_operator(numbers, operator):
    if operator is None:  # No operator, return the number itself
        return numbers[0]
    if operator == '+':
        return sum(numbers)
    elif operator == '-':
        return numbers[0] - sum(numbers[1:])
    elif operator == '*':
        result = 1
        for num in numbers:
            result *= num
        return result
    elif operator == '/':
        result = numbers[0]
        for num in numbers[1:]:
            if num == 0 or result % num != 0:  # Avoid division by zero and enforce divisibility
                return None
            result /= num
        return result
    return None

# Validate the result of a cage operation
def is_valid_cage_result(result, operator):
    return result is not None and result > 0

# Check if placing a number in a cell is valid
def is_valid(matrix, row, col, n, num, cages):
    # Check column uniqueness
    for r in range(row):
        if matrix[r][col] == num:
            return False

    # Check row uniqueness
    if num in matrix[row]:
        return False

    # Check cage constraints
    for cage, operator in cages:
        if (row, col) in cage:
            numbers_in_cage = [matrix[r][c] for r, c in cage if matrix[r][c] != 0]
            if num in numbers_in_cage:
                return False
            if len(numbers_in_cage) + 1 == len(cage):  # Cage is about to be full
                result = apply_operator(numbers_in_cage + [num], operator)
                if not is_valid_cage_result(result, operator):
                    return False
            break
    return True

# Solve the matrix with backtracking
def solve(matrix, n, row, col, cages):
    if row == n:
        return True
    if col == n:
        return solve(matrix, n, row + 1, 0, cages)

    numbers = list(range(1, n + 1))
    random.shuffle(numbers)

    for num in numbers:
        if is_valid(matrix, row, col, n, num, cages):
            matrix[row][col] = num
            if solve(matrix, n, row, col + 1, cages):
                return True
            matrix[row][col] = 0

    return False

# Generate a unique solution matrix
def generate_unique_matrix(n, cages):
    matrix = [[0] * n for _ in range(n)]
    if not solve(matrix, n, 0, 0, cages):
        raise ValueError("No solution found.")
    return matrix

# Get the cages with operators and their results
def get_cages_with_results(cages, matrix):
    cages_with_results = []
    for cage, operator in cages:
        # Extract the numbers in the cage from the solved matrix
        numbers_in_cage = [matrix[r][c] for r, c in cage]
        # Calculate the result of the operation for the cage
        result = apply_operator(numbers_in_cage, operator)
        cages_with_results.append([
         operator if operator is not None else '=',
       int(result),
           cage
        ])
    return cages_with_results

# Example usage
n =  ${parseInt(document.getElementById("sizeSelect").value)} # Grid size

# Generate random cages with operators
cages = generate_random_cages_with_operators(n)

# Create a matrix and solve it
matrix = generate_unique_matrix(n, cages)

# Get cages with results
cages_with_results = get_cages_with_results(cages, matrix)
# Print the cages with their results
newCages = json.dumps(cages_with_results)

newCages

`;

  const result = await pyodide.runPythonAsync(pythonCode);  
  generateBoard(result); 
}
