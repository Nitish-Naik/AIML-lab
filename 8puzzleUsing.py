import heapq

# Define the goal state
GOAL_STATE = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 0)  # 0 represents the empty tile
)

# Function to calculate the Manhattan distance
def manhattan_distance(state):
    distance = 0
    for row in range(3):
        for col in range(3):
            value = state[row][col]
            if value != 0:
                goal_row = (value - 1) // 3
                goal_col = (value - 1) % 3
                distance += abs(row - goal_row) + abs(col - goal_col)
    return distance

# Function to find the possible moves
def get_possible_moves(state):
    moves = []
    empty_row, empty_col = [(row, col) for row in range(3) for col in range(3) if state[row][col] == 0][0]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    for dr, dc in directions:
        new_row, new_col = empty_row + dr, empty_col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            # Create a new state by swapping the empty tile with the adjacent tile
            new_state = list(map(list, state))  # Deep copy the state
            new_state[empty_row][empty_col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[empty_row][empty_col]
            moves.append(tuple(map(tuple, new_state)))

    return moves

# A* search algorithm to solve the 8-puzzle
def a_star_search(start_state):
    open_set = []
    heapq.heappush(open_set, (0 + manhattan_distance(start_state), start_state, 0))  # (f, state, g)
    closed_set = set()
    came_from = {}

    while open_set:
        current_f, current_state, current_g = heapq.heappop(open_set)

        if current_state == GOAL_STATE:
            # Reconstruct the path
            path = []
            while current_state in came_from:
                path.append(current_state)
                current_state = came_from[current_state]
            path.append(current_state)  # Add the initial state at the end
            return path[::-1]  # Return reversed path to show the steps

        closed_set.add(current_state)

        for neighbor in get_possible_moves(current_state):
            if neighbor in closed_set:
                continue

            tentative_g = current_g + 1
            
            # Check if neighbor is already in open set
            if neighbor not in [state for f, state, g in open_set]:
                heapq.heappush(open_set, (tentative_g + manhattan_distance(neighbor), neighbor, tentative_g))
                came_from[neighbor] = current_state  # Keep track of the path

    return None  # No solution found

# Function to take dynamic input for the puzzle
def input_puzzle():
    print("Enter the initial state of the 8-puzzle (0 for empty space):")
    state = []
    entered_numbers = set()  # To ensure numbers from 0 to 8 are unique across the puzzle
    
    for i in range(3):
        while True:
            row = input(f"Enter row {i + 1} (3 numbers separated by spaces): ")
            try:
                numbers = list(map(int, row.split()))
                if len(numbers) == 3 and all(0 <= num <= 8 for num in numbers):
                    # Check if numbers are unique across the entire puzzle
                    if len(set(numbers)) < 3 or any(num in entered_numbers for num in numbers):
                        print("Duplicate numbers detected. Please enter unique numbers from 0 to 8.")
                    else:
                        state.append(tuple(numbers))
                        entered_numbers.update(numbers)  # Add the numbers to the set
                        break
                else:
                    print("Invalid input. Please enter 3 unique numbers between 0 and 8.")
            except ValueError:
                print("Invalid input. Please enter numbers only.")
    
    return tuple(state)

# Example usage
if __name__ == "__main__":
    # Get the initial state of the 8-puzzle from user input
    initial_state = input_puzzle()

    # Solve the puzzle
    solution_path = a_star_search(initial_state)

    if solution_path:
        print("\nSolution found! Steps to reach the goal:")
        for step in solution_path:
            for row in step:
                print(row)
            print()  # Newline for better readability
    else:
        print("No solution found.")
