import numpy as np
from pprint import pprint


# Example usage
board_size = 9  # Replace with actual board size
cur_state = np.random.rand(1, 4, board_size, board_size)  # Example current state
parts, positions = split_board_into_parts(cur_state)

# Verifying the results
for i, (part, pos) in enumerate(zip(parts, positions)):
    print(f"Part {i+1} shape: {part.shape}, Positions: {pos[:5]}...")  # Only printing the first 5 positions for brevity

