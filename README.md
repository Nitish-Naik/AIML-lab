# AIML-LAB


Sure, here’s a `README.md` file for your code:

---

# Puzzle Solver

This project implements a solution to a sliding puzzle problem using search algorithms. Specifically, it solves a 3x3 sliding puzzle (also known as the 8-puzzle) using Breadth-First Search (BFS) and Depth-First Search (DFS) techniques.

## Overview

The puzzle solver consists of the following components:
- **Node**: Represents a state in the search space.
- **StackFrontier**: A stack-based frontier for Depth-First Search (DFS).
- **QueueFrontier**: A queue-based frontier for Breadth-First Search (BFS).
- **Puzzle**: Encapsulates the puzzle logic, including state transitions, and the search algorithm.

## Files

- `puzzle_solver.py`: Contains the implementation of the puzzle solver.

## How It Works

### Classes

#### Node

The `Node` class represents a single state in the search space. It stores:
- `state`: The current state of the puzzle.
- `parent`: The parent node (previous state) that led to this state.
- `action`: The action taken to reach this state.

#### StackFrontier

The `StackFrontier` class implements a stack-based frontier (LIFO) used for Depth-First Search (DFS):
- `add(node)`: Adds a node to the frontier.
- `contains_state(state)`: Checks if the frontier contains a specific state.
- `empty()`: Checks if the frontier is empty.
- `remove()`: Removes and returns the last node added.

#### QueueFrontier

The `QueueFrontier` class inherits from `StackFrontier` but implements a queue-based frontier (FIFO) used for Breadth-First Search (BFS):
- `remove()`: Removes and returns the first node added.

#### Puzzle

The `Puzzle` class represents the puzzle and contains methods for solving it:
- `__init__(self, start, startIndex, goal, goalIndex)`: Initializes the puzzle with the start and goal states.
- `neighbors(self, state)`: Returns a list of neighboring states by moving the empty tile in all possible directions.
- `print(self)`: Prints the start state, goal state, number of states explored, and solution.
- `does_not_contain_state(self, state)`: Checks if a state is not already explored.
- `solve(self)`: Solves the puzzle using Breadth-First Search (BFS) and stores the solution.

### Example Usage

To solve a puzzle, initialize the `Puzzle` class with the start and goal states, and then call the `solve()` method followed by the `print()` method to display the results:

```python
import numpy as np
from puzzle_solver import Puzzle

start = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
goal = np.array([[2, 8, 1], [0, 4, 3], [7, 6, 5]])
startIndex = (1, 1)
goalIndex = (1, 0)

p = Puzzle(start, startIndex, goal, goalIndex)
p.solve()
p.print()
```

## Output

The output will show:
- The start state of the puzzle.
- The goal state of the puzzle.
- The number of states explored during the search.
- The sequence of actions and states leading from the start state to the goal state.

## Requirements

- Python 3.x
- NumPy library

## Running the Code

1. Ensure you have Python 3.x installed.
2. Install the NumPy library using `pip install numpy`.
3. Run the `puzzle_solver.py` script to see the puzzle solution.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to adjust any sections as needed, especially if you have specific requirements or additional information you want to include!