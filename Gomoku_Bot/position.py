from typing import List
from .config import config
# Checked

# Function to convert position to coordinate
def position2Coordinate(position: int, size: int) -> List[int]:
    return [position // size, position % size]

# Function to convert coordinate to position
def coordinate2Position(x: int, y: int, size: int) -> int:
    return x * size + y

# Check if points a and b are on the same line and the distance is less than maxDistance
def isLine(a: int, b: int, size: int) -> bool:
    maxDistance = config["inLineDistance"]
    [x1, y1] = position2Coordinate(a, size)
    [x2, y2] = position2Coordinate(b, size)
    return (
        (x1 == x2 and abs(y1 - y2) < maxDistance) or
        (y1 == y2 and abs(x1 - x2) < maxDistance) or
        (abs(x1 - x2) == abs(y1 - y2) and abs(x1 - x2) < maxDistance)
    )

# Check if all points in the array are on the same line as point p
def isAllInLine(p: int, arr: List[int], size: int) -> bool:
    for i in range(len(arr)):
        if not isLine(p, arr[i], size):
            return False
    return True

# Check if any point in the array is on the same line as point p
def hasInLine(p: int, arr: List[int], size: int) -> bool:
    for i in range(len(arr)):
        if isLine(p, arr[i], size):
            return True
    return False