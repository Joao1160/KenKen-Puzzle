function generateBoard(cagesObject) {
  const component = JSON.parse(cagesObject)

  localStorage.setItem('Cages', JSON.stringify(component[0][0]));
  localStorage.setItem('SolvedBoard', JSON.stringify(component[0][1]));
  localStorage.setItem('Steps', JSON.stringify(component[0][2]));

  const size = parseInt(document.getElementById("sizeSelect").value);

  cages = component[0][0]

  const container = document.getElementById("container");

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

function solve() {
  cages = JSON.parse(localStorage.getItem('Cages'));
  board = JSON.parse(localStorage.getItem('SolvedBoard'));
  codeSteps = JSON.parse(localStorage.getItem('Steps'));

  const container = document.getElementById("container");
  const size = parseInt(document.getElementById("sizeSelect").value);


  const containerSteps = document.getElementById("containerSteps");

  containerSteps.innerHTML = "";

  codeSteps.forEach((item, index) => {
    const Steps = document.createElement("div");
    Steps.className = "steps" ;
    Steps.style.gridTemplateColumns = `repeat(${size}, 1fr)`;
    Steps.style.gridTemplateRows = `repeat(${size}, 1fr)`;
    if(size >= 7 ){
        Steps.style.marginBottom = '150px'; // Replace '20px' with the desired value
    }
    item.forEach((item2, index2) => {
        item2.forEach((item3, index2) => {
        const squareItem = document.createElement("div");
        squareItem.className = "grid-square" ;
        squareItem.innerHTML = item3;
        Steps.appendChild(squareItem);
    });
});  
    containerSteps.appendChild(Steps);
});

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
                square.innerHTML = board[i][j];
              } else {
                if (cages[x][0] == "/") {
                  cages[x][0] = "÷";
                } else if (cages[x][0] == "*") {
                  cages[x][0] = "x";
                }
                square.innerHTML = `<span>${cages[x][1]}${cages[x][0]}</span>${board[i][j]}`;
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
        self.Steps = []

    def initialize(self):        
        self.N = ${parseInt(document.getElementById("sizeSelect").value)}
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
            index = index + 1 
        
        return NewCages
            

    def getBoard(self):
        return self.Board
    
    def getCages(self):
        return self.Cages
    

    def solve(self) -> bool:
        empty = self.findEmpty()
        if not empty:
            return True
        row, col = empty # => (x,y)
        for num in range(1, self.N + 1):
            if self.isValid(row, col, num):
                self.Board[row][col] = num
                #print(self.Board)
                self.Steps.append([row[:] for row in self.Board])  # Save the board before printing it
                if self.solve(): # recursive function
                    return True
                self.Board[row][col] = 0
                self.Steps.append([row[:] for row in self.Board])  # Save the board before printing it
                #print(self.Board)

        return False

    def returnResult(self):
        return(self.Steps)
        

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
            elif cage.opr == '-' :
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
    
    
K = KenKenSolver(0, 0)

CagesBoard = K.initialize()

if K.solve():
    solvedBoard =  K.getBoard()
else:
    solvedBoard = None

    
cages = []
for x in range(len(CagesBoard)):
    cages.append([CagesBoard[x][1], CagesBoard[x][2]])
    if isinstance(CagesBoard[x][0], list):  # Check if CagesBoard[x][2] is a list
        cage = []
        for y in range(len(CagesBoard[x][0])):
            cage.append([CagesBoard[x][0][y][0], CagesBoard[x][0][y][1]])
        cages[x].append(cage)

component = []
steps = K.returnResult()
component.append([cages, solvedBoard , steps])
newCages = json.dumps(component)
newCages
`;

  const result = await pyodide.runPythonAsync(pythonCode);    
  generateBoard(result); 
}
