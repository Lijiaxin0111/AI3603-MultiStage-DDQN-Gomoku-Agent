import random
# Checked
class ZobristCache:
    def __init__(self, size):
        self.size = size
        self.zobristTable = self.initializeZobristTable(size)
        self.hash = 0

    def initializeZobristTable(self, size):
        table = []
        for i in range(size):
            table.append([])
            for j in range(size):
                table[i].append({
                    1: random.getrandbits(64),  # black
                    -1: random.getrandbits(64)  # white
                })
        return table

    def togglePiece(self, x, y, role):
        self.hash ^= self.zobristTable[x][y][role]

    def getHash(self):
        return self.hash

if __name__ == '__main__':
    # Example usage
    size = 8
    cache = ZobristCache(size)
    x = 3
    y = 4
    role = 1
    cache.togglePiece(x, y, role)
    hash_value = cache.getHash()
    print(hash_value)