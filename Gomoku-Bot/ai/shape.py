
import re

# Define patterns using regular expressions
patterns = {
    'five': re.compile('11111'),
    'block_five': re.compile('211111|111112'),
    'four': re.compile('011110'),
    'block_four': re.compile('10111|11011|11101|211110|211101|211011|210111|011112|101112|110112|111012'),
    'three': re.compile('011100|011010|010110|001110'),
    'block_three': re.compile('211100|211010|210110|001112|010112|011012'),
    'two': re.compile('001100|011000|000110|010100|001010'),
}

# Define shapes with associated scores
shapes = {
    'FIVE': 5,
    'BLOCK_FIVE': 50,
    'FOUR': 4,
    'FOUR_FOUR': 44,  # Double four
    'FOUR_THREE': 43,  # Four with an open three
    'THREE_THREE': 33,  # Double three
    'BLOCK_FOUR': 40,
    'THREE': 3,
    'BLOCK_THREE': 30,
    'TWO_TWO': 22,  # Double two
    'TWO': 2,
    'NONE': 0
}

# Initialize a performance record
performance = {
    'five': 0,
    'block_five': 0,
    'four': 0,
    'block_four': 0,
    'three': 0,
    'block_three': 0,
    'two': 0,
    'none': 0,
    'total': 0
}

# Function to detect shapes on the board
def get_shape(board, x, y, offset_x, offset_y, role):
    """
    Detect shape at a given board position.
    :param board: The game board.
    :param x: X-coordinate.
    :param y: Y-coordinate.
    :param offset_x: X-direction offset for scanning.
    :param offset_y: Y-direction offset for scanning.
    :param role: Current player's role.
    :return: A tuple of shape, self count, opponent count, and empty count.
    """
    opponent = -role
    empty_count = 0
    self_count = 1
    opponent_count = 0
    shape = shapes['NONE']

    # Skip empty nodes
    if (
        board[x + offset_x + 1][y + offset_y + 1] == 0
        and board[x - offset_x + 1][y - offset_y + 1] == 0
        and board[x + 2 * offset_x + 1][y + 2 * offset_y + 1] == 0
        and board[x - 2 * offset_x + 1][y - 2 * offset_y + 1] == 0
    ):
        return [0, self_count, opponent_count, empty_count]
    # Check for 'two' pattern
    for i in range(-3, 4):
        if i == 0:
            continue
        nx, ny = x + i * offset_x, y + i * offset_y
        current_role = board.get((nx, ny))
        if current_role is None:
            continue
        if current_role == 2:
            opponent_count += 1
        elif current_role == role:
            self_count += 1
        elif current_role == 0:
            empty_count += 1

    if self_count == 2:
        if opponent_count == 0:
            return shapes['TWO'], self_count, opponent_count, empty_count
        else:
            return shapes['NONE'], self_count, opponent_count, empty_count

    # Reset counts and prepare string for pattern matching
    empty_count, self_count, opponent_count = 0, 1, 0
    result_string = '1'

    # Build result string for pattern matching
    for i in range(1, 6):
        nx = x + i * offset_x + 1
        ny = y + i * offset_y + 1
        currentRole = board[nx][ny]
        if currentRole == 2:
            result_string += '2'
        elif currentRole == 0:
            result_string += '0'
        else:
            result_string += '1' if currentRole == role else '2'
        if currentRole == 2 or currentRole == opponent:
            opponent_count += 1
            break
        if currentRole == 0:
            empty_count += 1
        if currentRole == role:
            self_count += 1
    
    for i in range(1, 6):
        nx = x - i * offset_x + 1
        ny = y - i * offset_y + 1
        currentRole = board[nx][ny]
        if currentRole == 2:
            result_string = '2' + result_string
        elif currentRole == 0:
            result_string = '0' + result_string
        else:
            result_string = '1' if currentRole == role else '2' + result_string
        if currentRole == 2 or currentRole == opponent:
            opponent_count += 1
            break
        if currentRole == 0:
            empty_count += 1
        if currentRole == role:
            self_count += 1

    # Check patterns and update performance
    for pattern_key, shape_key in [('five', 'FIVE'), ('four', 'FOUR'), ('block_four', 'BLOCK_FOUR'),
                                   ('three', 'THREE'), ('block_three', 'BLOCK_THREE'), ('two', 'TWO')]:
        if patterns[pattern_key].search(result_string):
            shape = shapes[shape_key]
            performance[pattern_key] += 1
            performance['total'] += 1
            break
    ## 尽量减少多余字符串生成
    if self_count <= 1 or len(result_string) < 5:
        return shape, self_count, opponent_count, empty_count

    return shape, self_count, opponent_count, empty_count

def count_shape(board, x, y, offset_x, offset_y, role):
    opponent = - role

    inner_empty_count = 0  # Number of empty positions inside the player's stones
    temp_empty_count = 0
    self_count = 0  # Number of the player's stones in the shape
    total_length = 0

    side_empty_count = 0  # Number of empty positions on the side of the shape
    no_empty_self_count = 0
    one_empty_self_count = 0

    # Right direction
    for i in range(1, 6):
        nx = x + i * offset_x + 1
        ny = y + i * offset_y + 1
        current_role = board[nx][ny]
        if current_role == 2 or current_role == opponent:
            break
        if current_role == role:
            self_count += 1
            side_empty_count = 0
            if temp_empty_count:
                inner_empty_count += temp_empty_count
                temp_empty_count = 0
            if inner_empty_count == 0:
                no_empty_self_count += 1
                one_empty_self_count += 1
            elif inner_empty_count == 1:
                one_empty_self_count += 1
        total_length += 1
        if current_role == 0:
            temp_empty_count += 1
            side_empty_count += 1
        if side_empty_count >= 2:
            break

    if not inner_empty_count:
        one_empty_self_count = 0

    return {
        'self_count': self_count,
        'total_length': total_length,
        'no_empty_self_count': no_empty_self_count,
        'one_empty_self_count': one_empty_self_count,
        'inner_empty_count': inner_empty_count,
        'side_empty_count': side_empty_count
    }

# Fast shape detection function
def get_shape_fast(board, x, y, offsetX, offsetY, role):
    if (
        board[x + offsetX + 1][y + offsetY + 1] == 0
        and board[x - offsetX + 1][y - offsetY + 1] == 0
        and board[x + 2 * offsetX + 1][y + 2 * offsetY + 1] == 0
        and board[x - 2 * offsetX + 1][y - 2 * offsetY + 1] == 0
    ):
        return [shapes['NONE'], 1]

    selfCount = 1
    totalLength = 1
    shape = shapes['NONE']

    leftEmpty = 0
    rightEmpty = 0
    noEmptySelfCount = 1
    OneEmptySelfCount = 1

    left = count_shape(board, x, y, -offsetX, -offsetY, role)
    right = count_shape(board, x, y, offsetX, offsetY, role)

    selfCount = left['self_count'] + right['self_count'] + 1
    totalLength = left['total_length'] + right['total_length'] + 1
    noEmptySelfCount = left['no_empty_self_count'] + right['no_empty_self_count'] + 1
    OneEmptySelfCount = max(
        left['one_empty_self_count'] + right['no_empty_self_count'],
        left['no_empty_self_count'] + right['one_empty_self_count'],
    ) + 1
    rightEmpty = right['side_empty_count']
    leftEmpty = left['side_empty_count']

    if totalLength < 5:
        return [shape, selfCount]

    if noEmptySelfCount >= 5:
        if rightEmpty > 0 and leftEmpty > 0:
            return [shapes['FIVE'], selfCount]
        else:
            return [shapes['BLOCK_FIVE'], selfCount]

    if noEmptySelfCount == 4:
        if (
            (rightEmpty >= 1 or right['one_empty_self_count'] > right['no_empty_self_count'])
            and (leftEmpty >= 1 or left['one_empty_self_count'] > left['no_empty_self_count'])
        ):
            return [shapes['FOUR'], selfCount]
        elif not (rightEmpty == 0 and leftEmpty == 0):
            return [shapes['BLOCK_FOUR'], selfCount]

    if OneEmptySelfCount == 4:
        return [shapes['BLOCK_FOUR'], selfCount]

    if noEmptySelfCount == 3:
        if (rightEmpty >= 2 and leftEmpty >= 1) or (rightEmpty >= 1 and leftEmpty >= 2):
            return [shapes['THREE'], selfCount]
        else:
            return [shapes['BLOCK_THREE'], selfCount]

    if OneEmptySelfCount == 3:
        if rightEmpty >= 1 and leftEmpty >= 1:
            return [shapes['THREE'], selfCount]
        else:
            return [shapes['BLOCK_THREE'], selfCount]

    if (noEmptySelfCount == 2 or OneEmptySelfCount == 2) and totalLength > 5:
        shape = shapes['TWO']

    return [shape, selfCount]

# Helper functions to check for specific shapes
def is_five(shape):
    # Checked
    return shape in [shapes['FIVE'], shapes['BLOCK_FIVE']]

def is_four(shape):
    # Checked
    return shape in [shapes['FOUR'], shapes['BLOCK_FOUR']]

# Function to get all shapes at a specific point
def get_all_shapes_of_point(shape_cache, x, y, role = None):
    # Checked
    roles = [role] if role else [1, -1]
    result = []
    for r in roles:
        for d in range(4):
            shape = shape_cache[r][d][x][y]
            if shape > 0:
                result.append(shape)
    return result


if __name__ == "__main__":
    pass