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
            cage,
            operator if operator is not None else '=',
            int(result)
        ])
    return cages_with_results

class Location:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class Cage:
    def __init__(self, goal: int, opr: str, squares: list):
        self.goal = goal
        self.opr = opr
        self.squares = squares  # List of Location objects [(1,2),(1,3),(2,2)]

class KenKenSolver:
    def __init__(self, N: int = 0, NCage: int = 0):
        self.N = N  # Size of the board (N x N)
        self.NCage = NCage  # Number of cages
        self.Board = []  # List of numbers in board
        self.CageBoard = []  # List of IDs Cages in Board 
        self.Cages = []  # List of Cage objects

    def initialize(self):        
        self.N = int(input("Enter the size of the board (N): "))
        cages = generate_random_cages_with_operators(self.N)
        matrix = generate_unique_matrix(self.N, cages)
        NewCages = get_cages_with_results(cages, matrix)
        self.NCage = 0
        for row in NewCages : 
            self.NCage = self.NCage + 1

        # Resize the board, cage board, and cages list
        self.Board = [[0 for x in range(self.N)] for y in range(self.N)]
        self.CageBoard = [[0 for x in range(self.N)] for y in range(self.N)]
        self.Cages = [Cage(0, '', []) for _ in range(self.NCage)] # 0 for Goal , '' for operation , [] for list of location of every cage
        
        index = 0
        for item in NewCages:
            for loc in item[0]: 
                self.CageBoard[loc[0]][loc[1]] = index
                self.Cages[index].squares.append(loc)
            self.Cages[index].goal = item[2]
            self.Cages[index].opr = item[1]
            print(f"ID = {index} , Operation = '{self.Cages[index].opr}' , Goal = {self.Cages[index].goal}")
            index = index + 1
                
        # print list of Cages IDs
        print("\nCagesBoard IDs")
        for row in self.CageBoard: 
            print(" ".join(map(str, row)))
            
 

    def printResult(self):
        # Print the resulting board
        for row in self.Board: # row represent as [ 1 , 2 , 3 ]
            print(" ".join(map(str, row)))

    def printState(self, row, col, num, action):
        print(f"\n{action}: Placing {num} at ({row + 1}, {col + 1})")
        for r in self.Board:
            print(" ".join(map(str, r)))
        print("\n")

    def solve(self) -> bool:
        empty = self.findEmpty()
        if not empty:
            return True
        row, col = empty # => (x,y)
        for num in range(1, self.N + 1):
            if self.isValid(row, col, num):
                self.Board[row][col] = num
                #self.printState(row, col, num, "TRYING")
                if self.solve(): # recursive function
                    return True
                self.Board[row][col] = 0
                #self.printState(row, col, num, "BACKTRACKING")
        return False

    def isValid(self, row: int, col: int, num: int) -> bool:
        return (
            self.checkRow(row, num) and # check if num exist in row
            self.checkCol(col, num) and # check if num exist in col
            self.checkCage(row, col, num) # check if num achieves the operation
        )

    def checkRow(self, row: int, num: int) -> bool:
        return num not in self.Board[row]

    def checkCol(self, col: int, num: int) -> bool:
        return num not in [self.Board[row][col] for row in range(self.N)]

    def checkCage(self, row: int, col: int, num: int) -> bool:
        cage_id = self.CageBoard[row][col]
        cage = self.Cages[cage_id]
        # temp_board = self.Board[row][col]  # save the number in temp it always be 0
        self.Board[row][col] = num

        values = [self.Board[loc[0]][loc[1]] for loc in cage.squares if self.Board[loc[0]][loc[1]] != 0]

        if len(values) == len(cage.squares):
            if cage.opr == '+':
                if sum(values) != cage.goal:
                    self.Board[row][col] = 0 
                    return False
            elif cage.opr == '-':
                if abs(values[0] - values[1]) != cage.goal:
                    self.Board[row][col] = 0
                    return False
            elif cage.opr == '*':
                product = 1
                for x in values:
                    product *= x
                if product != cage.goal:
                    self.Board[row][col] = 0
                    return False
            elif cage.opr == '/':
                if max(values) / min(values) != cage.goal:
                    self.Board[row][col] = 0
                    return False
            elif cage.opr == '=':
                if values[0] != cage.goal:
                    self.Board[row][col] = 0
                    return False

        self.Board[row][col] = 0 
        return True

    def findEmpty(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.Board[i][j] == 0:
                    return i, j
        return None

# Main program
if __name__ == "__main__":
    
    K = KenKenSolver(0, 0)
    K.initialize()
    if K.solve():
        print("\nSolved Board:")
        K.printResult()
    else:
        print("Oops!")

