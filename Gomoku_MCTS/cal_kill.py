import numpy as np

def find_live_four_completion(board_class, player):
    board_size = board_class.width
    state = board_class.states
    board = np.zeros((board_size, board_size))
    for move in state:
        board[move // board_size, move % board_size] = state[move]

    def is_valid_completion(x, y, dx, dy):
        # Check if placing a stone at (x, y) completes a live four
        if 0 <= x < board_size and 0 <= y < board_size and board[x, y] == 0:
            count = 0
            for i in range(1, 6):  # Check five positions in one direction
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    if board[nx, ny] == player:
                        count += 1
                    else:
                        break

            for i in range(1, 6):  # Check five positions in the opposite direction
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    if board[nx, ny] == player:
                        count += 1
                    else:
                        break

            if count == 4:  # Exactly four stones found around the empty spot
                return True

        return False

    # Check all directions for a possible completion of live four
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonal_down, diagonal_up
    for x in range(board_size):
        for y in range(board_size):
            for dx, dy in directions:
                if is_valid_completion(x, y, dx, dy):
                    return x * board_size + y
    return None



if __name__ == "__main__":


    # Example usage
    board_example = np.zeros((15, 15), dtype=int)
    board_example[1,3] = 1  # Create a horizontal live four for player 1 (black)
    board_example[2, 2] = 1
    board_example[3, 1] = 1
    board_example[4, 0] = 1
    # Find the position to complete live four into five for player 1 (black)
    completion_move = find_live_four_completion(board_example, 1)
    print(completion_move)