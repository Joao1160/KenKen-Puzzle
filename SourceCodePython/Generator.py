import random

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
if __name__ == "__main__":
    n =  5# Grid size

    # Generate random cages with operators
    cages = generate_random_cages_with_operators(n)
    
    # Create a matrix and solve it
    matrix = generate_unique_matrix(n, cages)

    # Get cages with results
    cages_with_results = get_cages_with_results(cages, matrix)
    # Print the cages with their results
    print( cages_with_results)

