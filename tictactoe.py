def print_board(board):
    """Prints the current game board."""
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def check_winner(board):
    """Checks for a winner. Returns 'X', 'O' if there's a winner, or None if no winner yet."""
    # Check rows, columns, and diagonals
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != " ":
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != " ":
            return board[0][i]
    
    if board[0][0] == board[1][1] == board[2][2] != " ":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != " ":
        return board[0][2]
    
    return None  # No winner

def is_board_full(board):
    """Checks if the board is full."""
    return all(cell != " " for row in board for cell in row)

def minimax(board, depth, is_maximizing):
    """Minimax algorithm to find the best move for the AI."""
    winner = check_winner(board)
    if winner == "X":  # Player wins
        return -1
    if winner == "O":  # AI wins
        return 1
    if is_board_full(board):  # Tie
        return 0

    if is_maximizing:
        best_score = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "O"  # AI move
                    score = minimax(board, depth + 1, False)
                    board[i][j] = " "  # Undo move
                    best_score = max(best_score, score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "X"  # Player move
                    score = minimax(board, depth + 1, True)
                    board[i][j] = " "  # Undo move
                    best_score = min(best_score, score)
        return best_score

def best_move(board):
    """Finds the best move for the AI using the minimax algorithm."""
    move = (-1, -1)
    best_score = float('-inf')
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board[i][j] = "O"  # AI move
                score = minimax(board, 0, False)
                board[i][j] = " "  # Undo move
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

def tic_tac_toe():
    """Main function to run the Tic Tac Toe game."""
    board = [[" " for _ in range(3)] for _ in range(3)]
    print("Welcome to Tic Tac Toe!")
    print_board(board)

    while True:
        # Player's turn
        while True:
            try:
                row, col = map(int, input("Enter your move (row and column from 0 to 2, separated by space): ").split())
                if board[row][col] == " ":
                    board[row][col] = "X"  # Player move
                    break
                else:
                    print("Cell already taken! Choose another.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter row and column as two numbers (0-2).")

        print_board(board)
        if check_winner(board) == "X":
            print("Congratulations! You win!")
            break
        if is_board_full(board):
            print("It's a tie!")
            break

        # AI's turn
        print("AI's turn:")
        ai_move = best_move(board)
        if ai_move != (-1, -1):
            board[ai_move[0]][ai_move[1]] = "O"  # AI move
            print_board(board)
            if check_winner(board) == "O":
                print("AI wins! Better luck next time.")
                break
            if is_board_full(board):
                print("It's a tie!")
                break

if __name__ == "__main__":
    tic_tac_toe()
