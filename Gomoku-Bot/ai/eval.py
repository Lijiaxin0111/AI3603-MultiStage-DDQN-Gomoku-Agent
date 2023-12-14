from enum import Enum
import math
from shape import shapes, get_shape_fast, is_five, is_four, get_all_shapes_of_point
from position import coordinate2Position, isLine, isAllInLine, hasInLine, position2Coordinate
from config import config
from datetime import datetime
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import torch
from minimax_Net import BoardEvaluationNet as net

mini_max_net = net(board_size=15)
mini_max_net.load_state_dict(torch.load(os.path.join(dir_path, 'train_data/model', 'best_loss=609.3356355479785.pth')))
mini_max_net.eval()


# Enum to represent different shapes
class Shapes(Enum):
    FIVE = 0
    BLOCK_FIVE = 1
    FOUR = 2
    FOUR_FOUR = 3
    FOUR_THREE = 4
    THREE_THREE = 5
    BLOCK_FOUR = 6
    THREE = 7
    BLOCK_THREE = 8
    TWO_TWO = 9
    TWO = 10
    NONE = 11


# Constants representing scores for each shape
FIVE = 10000000
BLOCK_FIVE = FIVE
FOUR = 100000
FOUR_FOUR = FOUR  # 双冲四
FOUR_THREE = FOUR  # 冲四活三
THREE_THREE = FOUR / 2  # 双活三
BLOCK_FOUR = 1500
THREE = 1000
BLOCK_THREE = 150
TWO_TWO = 200  # 双活二
TWO = 100
BLOCK_TWO = 15
ONE = 10
BLOCK_ONE = 1


# Function to calculate the real shape score based on the shape
def getRealShapeScore(shape: Shapes) -> int:
    # Checked
    if shape == shapes['FIVE']:
        return FOUR
    elif shape == shapes['BLOCK_FIVE']:
        return BLOCK_FOUR
    elif shape in [shapes['FOUR'], shapes['FOUR_FOUR'], shapes['FOUR_THREE']]:
        return THREE
    elif shape == shapes['BLOCK_FOUR']:
        return BLOCK_THREE
    elif shape == shapes['THREE']:
        return TWO
    elif shape == shapes['THREE_THREE']:
        return math.floor(THREE_THREE / 10)
    elif shape == shapes['BLOCK_THREE']:
        return BLOCK_TWO
    elif shape == shapes['TWO']:
        return ONE
    elif shape == shapes['TWO_TWO']:
        return math.floor(TWO_TWO / 10)
    else:
        return 0


# List of all directions
allDirections = [
    [0, 1],  # Horizontal
    [1, 0],  # Vertical
    [1, 1],  # Diagonal \
    [1, -1]  # Diagonal /
]


# Function to get the index of a direction
def direction2index(ox: int, oy: int) -> int:
    # Checked
    if ox == 0:
        return 0  # |
    elif oy == 0:
        return 1  # -
    elif ox == oy:
        return 2  # \
    elif ox != oy:
        return 3  # /


# Performance dictionary
performance = {
    "updateTime": 0,
    "getPointsTime": 0
}


class Evaluate:
    def __init__(self, size=15):
        # Checked
        self.size = size
        self.board = [[2] * (size + 2) for _ in range(size + 2)]
        for i in range(size + 2):
            for j in range(size + 2):
                if i == 0 or j == 0 or i == self.size + 1 or j == self.size + 1:
                    self.board[i][j] = 2
                else:
                    self.board[i][j] = 0
        self.blackScores = [[0] * self.size for _ in range(size)]
        self.whiteScores = [[0] * self.size for _ in range(size)]
        self.initPoints()
        self.history = []  # List of [position, role]

    def move(self, x, y, role):
        # Checked
        # Clear the cache first
        for d in [0, 1, 2, 3]:
            self.shapeCache[role][d][x][y] = 0
            self.shapeCache[-role][d][x][y] = 0
        self.blackScores[x][y] = 0
        self.whiteScores[x][y] = 0
        # Update the board
        self.board[x + 1][y + 1] = role  ## Adjust for the added wall
        self.updatePoint(x, y)
        self.history.append([coordinate2Position(x, y, self.size), role])

    def undo(self, x, y):
        # Checked
        self.board[x + 1][y + 1] = 0
        self.updatePoint(x, y)
        self.history.pop()

    def initPoints(self):
        # Checked
        # Initialize the cache, avoid calculating the same points multiple times
        self.shapeCache = {}
        for role in [1, -1]:
            self.shapeCache[role] = {}
            for direction in [0, 1, 2, 3]:
                self.shapeCache[role][direction] = [[0] * self.size for _ in range(self.size)]

        self.pointsCache = {}
        for role in [1, -1]:
            self.pointsCache[role] = {}
            for shape in shapes:
                self.pointsCache[role][shape] = set()

    def getPointsInLine(self, role):
        # Checked
        pointsInLine = {}
        hasPointsInLine = False
        for key in shapes:
            pointsInLine[shapes[key]] = set()

        last2Points = [position for position, role in self.history[-config['inlineCount']:]]
        processed = {}
        # 在last2Points中查找是否有点位在一条线上
        for r in [role, -role]:
            for point in last2Points:
                x, y = position2Coordinate(point, self.size)
                for ox, oy in allDirections:
                    for sign in [1, -1]:
                        for step in range(1, config['inLineDistance'] + 1):
                            nx = x + sign * step * ox
                            ny = y + sign * step * oy
                            position = coordinate2Position(nx, ny, self.size)
                            # 检测是否到达边界
                            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
                                break
                            if self.board[nx + 1][ny + 1] != 0:
                                continue
                            if processed.get(position) == r:
                                continue
                            processed[position] = r
                            for direction in [0, 1, 2, 3]:
                                shape = self.shapeCache[r][direction][nx][ny]
                                # 到达边界停止，但是注意到达对方棋子不能停止
                                if shape:
                                    pointsInLine[shape].add(coordinate2Position(nx, ny, self.size))
                                    hasPointsInLine = True

        if hasPointsInLine:
            return pointsInLine
        return False

    def getPoints(self, role, depth, vct, vcf):
        first = role if depth % 2 == 0 else -role  # 先手
        start = datetime.now()

        if config['onlyInLine'] and len(self.history) >= config['inlineCount']:
            points_in_line = self.getPointsInLine(role)
            if points_in_line:
                performance['getPointsTime'] += (datetime.now() - start).total_seconds()
                return points_in_line

        points = {}  # 全部点位

        for key in shapes.keys():
            points[shapes[key]] = set()

        last_points = [position for position, _ in self.history[-4:]]

        for r in [role, -role]:
            # 这里是直接遍历了这个棋盘上的所有点位，如果棋盘很大，这里会有性能问题；可以用神经网络来预测
            for i in range(self.size):
                for j in range(self.size):
                    four_count = 0
                    block_four_count = 0
                    three_count = 0

                    for direction in [0, 1, 2, 3]:
                        if self.board[i + 1][j + 1] != 0:
                            continue

                        shape = self.shapeCache[r][direction][i][j]

                        if not shape:
                            continue

                        point = i * self.size + j

                        if vcf:
                            if r == first and not is_four(shape) and not is_five(shape):
                                continue
                            if r == -first and is_five(shape):
                                continue

                        if vct:
                            if depth % 2 == 0:
                                if depth == 0 and r != first:
                                    continue
                                if shape != shapes['THREE'] and not is_four(shape) and not is_five(shape):
                                    continue
                                if shape == shapes['THREE'] and r != first:
                                    continue
                                if depth == 0 and r != first:
                                    continue
                                if depth > 0:
                                    if shape == shapes['THREE'] and len(
                                            get_all_shapes_of_point(self.shapeCache, i, j, r)) == 1:
                                        continue
                                    if shape == shapes['BLOCK_FOUR'] and len(
                                            get_all_shapes_of_point(self.shapeCache, i, j, r)) == 1:
                                        continue
                            else:
                                if shape != shapes['THREE'] and not is_four(shape) and not is_five(shape):
                                    continue
                                if shape == shapes['THREE'] and r == -first:
                                    continue
                                if depth > 1:
                                    if shape == shapes['BLOCK_FOUR'] and len(
                                            get_all_shapes_of_point(self.shapeCache, i, j)) == 1:
                                        continue
                                    if shape == shapes['BLOCK_FOUR'] and not hasInLine(point, last_points, self.size):
                                        continue

                        if vcf:
                            if not is_four(shape) and not is_five(shape):
                                continue

                        if depth > 2 and (shape == shapes['TWO'] or shape == shapes['TWO_TWO'] or shape == shapes[
                            'BLOCK_THREE']) and not hasInLine(point, last_points, self.size):
                            continue

                        points[shape].add(point)

                        if shape == shapes['FOUR']:
                            four_count += 1
                        elif shape == shapes['BLOCK_FOUR']:
                            block_four_count += 1
                        elif shape == shapes['THREE']:
                            three_count += 1

                        union_shape = None

                        if four_count >= 2:
                            union_shape = shapes['FOUR_FOUR']
                        elif block_four_count and three_count:
                            union_shape = shapes['FOUR_THREE']
                        elif three_count >= 2:
                            union_shape = shapes['THREE_THREE']

                        if union_shape:
                            points[union_shape].add(point)

        performance['getPointsTime'] += (datetime.now() - start).total_seconds()

        return points

    """
      当一个位置发生变时候，要更新这个位置的四个方向上得分，更新规则是：
        1. 如果这个位置是空的，那么就重新计算这个位置的得分
        2. 如果碰到了边界或者对方的棋子，那么就停止计算
        3. 如果超过2个空位，那么就停止计算
        4. 要更新自己的和对方的得分
    """

    def updatePoint(self, x, y):
        # Checked
        start = datetime.now()
        self.updateSinglePoint(x, y, 1)
        self.updateSinglePoint(x, y, -1)

        for ox, oy in allDirections:
            for sign in [1, -1]:  # -1 for negative direction, 1 for positive direction
                for step in range(1, 6):
                    reachEdge = False
                    for role in [1, -1]:
                        nx = x + sign * step * ox + 1  # +1 to adjust for wall
                        ny = y + sign * step * oy + 1  # +1 to adjust for wall
                        # Stop if wall or opponent's piece is found
                        if self.board[nx][ny] == 2:
                            reachEdge = True
                            break
                        elif self.board[nx][ny] == -role:  # Change role if opponent's piece is found
                            continue
                        elif self.board[nx][ny] == 0:
                            self.updateSinglePoint(nx - 1, ny - 1, role,
                                                   [sign * ox, sign * oy])  # -1 to adjust back from wall
                    if reachEdge:
                        break
        performance['updateTime'] += (datetime.now() - start).total_seconds()

    """
       计算单个点的得分
       计算原理是：
       在当前位置放一个当前角色的棋子，遍历四个方向，生成四个方向上的字符串，用patters来匹配字符串, 匹配到的话，就将对应的得分加到scores上
       四个方向的字符串生成规则是：向两边都延伸5个位置，如果遇到边界或者对方的棋子，就停止延伸
       在更新周围棋子时，只有一个方向需要更新，因此可以传入direction参数，只更新一个方向
    """

    def updateSinglePoint(self, x, y, role, direction=None):
        # Checked
        if self.board[x + 1][y + 1] != 0:
            return  # Not an empty spot

        # Temporarily place the piece
        self.board[x + 1][y + 1] = role

        directions = []
        if direction:
            directions.append(direction)
        else:
            directions = allDirections

        shapeCache = self.shapeCache[role]

        # Clear the cache first
        for ox, oy in directions:
            shapeCache[direction2index(ox, oy)][x][y] = shapes['NONE']

        score = 0
        blockFourCount = 0
        threeCount = 0
        twoCount = 0

        # Calculate existing score
        for intDirection in [0, 1, 2, 3]:
            shape = shapeCache[intDirection][x][y]
            if shape > shapes['NONE']:
                score += getRealShapeScore(shape)
                if shape == shapes['BLOCK_FOUR']:
                    blockFourCount += 1
                if shape == shapes['THREE']:
                    threeCount += 1
                if shape == shapes['TWO']:
                    twoCount += 1

        for ox, oy in directions:
            intDirection = direction2index(ox, oy)
            shape, selfCount = get_shape_fast(self.board, x, y, ox, oy, role)
            if not shape:
                continue
            if shape:
                # Note: Only cache single shapes, do not cache compound shapes like double threes, as they depend on two shapes
                shapeCache[intDirection][x][y] = shape
                if shape == shapes['BLOCK_FOUR']:
                    blockFourCount += 1
                if shape == shapes['THREE']:
                    threeCount += 1
                if shape == shapes['TWO']:
                    twoCount += 1
                if blockFourCount >= 2:
                    shape = shapes['FOUR_FOUR']
                elif blockFourCount and threeCount:
                    shape = shapes['FOUR_THREE']
                elif threeCount >= 2:
                    shape = shapes['THREE_THREE']
                elif twoCount >= 2:
                    shape = shapes['TWO_TWO']
                score += getRealShapeScore(shape)

        self.board[x + 1][y + 1] = 0  # Remove the temporary piece

        if role == 1:
            self.blackScores[x][y] = score
        else:
            self.whiteScores[x][y] = score

        return score

    def evaluate(self, role):
        # Checked
        blackScore = 0
        whiteScore = 0

        for i in range(len(self.blackScores)):
            for j in range(len(self.blackScores[i])):
                blackScore += self.blackScores[i][j]

        for i in range(len(self.whiteScores)):
            for j in range(len(self.whiteScores[i])):
                whiteScore += self.whiteScores[i][j]

        score = blackScore - whiteScore if role == 1 else whiteScore - blackScore
        return score

    def getMoves(self, role, depth, onThree=False, onlyFour=False, use_net = False):
        # Checked
        train_data = 0
        if use_net and role == 1:
            value_move_num = 6
            input = torch.Tensor(np.array(self.board)[1:-1, 1:-1]).unsqueeze(0)
            scores = mini_max_net(input)
            flattened_scores = scores.flatten()

            moves = (flattened_scores.argsort(descending=True)[:value_move_num]).tolist()
            # print(moves)
        else:
            moves, model_train_maxtrix = self._getMoves(role, depth, onThree, onlyFour)
            train_data = {"state": np.array(self.board)[1:-1, 1:-1], "scores": model_train_maxtrix}
        moves = [(move // self.size, move % self.size) for move in moves]
        # cut the self.board into normal size
        print("moves", moves)

        return moves, train_data

    def _getMoves(self, role, depth, only_three=False, only_four=False):
        """
        Get possible moves based on the current game state.
        """
        points = self.getPoints(role, depth, only_three, only_four)
        fives = points[shapes['FIVE']]
        block_fives = points[shapes['BLOCK_FIVE']]

        # To train the model, we need to get all these points's score and store it to board size matrix
        # Then we can use this matrix to train the model, given a state, we want it to output the score of each point, then we can choose the highest score point
        model_train_matrix = [[0] * self.size for _ in range(self.size)]

        if fives and len(fives) > 0 or block_fives and len(block_fives) > 0:
            for point in fives:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(FIVE, model_train_matrix[x][y])
            for point in block_fives:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(BLOCK_FIVE, model_train_matrix[x][y])

            return set(list(fives) + list(block_fives)), model_train_matrix

        fours = points[shapes['FOUR']]
        block_fours = points[shapes['BLOCK_FOUR']]  # Block four is special, consider it in both four and three
        if only_four or (fours and len(fours) > 0):
            for point in fours:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(FOUR, model_train_matrix[x][y])

            for point in block_fours:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(BLOCK_FOUR, model_train_matrix[x][y])

            return set(list(fours) + list(block_fours)), model_train_matrix

        four_fours = points[shapes['FOUR_FOUR']]
        if four_fours and len(four_fours) > 0:
            for point in four_fours:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(FOUR_FOUR, model_train_matrix[x][y])

            for point in block_fours:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(BLOCK_FOUR, model_train_matrix[x][y])

            return set(list(four_fours) + list(block_fours)), model_train_matrix

        # Double threes and active threes
        threes = points[shapes['THREE']]
        four_threes = points[shapes['FOUR_THREE']]
        if four_threes and len(four_threes) > 0:
            for point in four_threes:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(FOUR_THREE, model_train_matrix[x][y])

            for point in block_fours:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(BLOCK_FOUR, model_train_matrix[x][y])

            for point in threes:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(THREE, model_train_matrix[x][y])

            return set(list(four_threes) + list(block_fours) + list(threes)), model_train_matrix

        three_threes = points[shapes['THREE_THREE']]
        if three_threes and len(three_threes) > 0:

            for point in three_threes:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(THREE_THREE, model_train_matrix[x][y])

            for point in block_fours:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(BLOCK_FOUR, model_train_matrix[x][y])

            for point in threes:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(THREE, model_train_matrix[x][y])

            return set(list(three_threes) + list(block_fours) + list(threes)), model_train_matrix

        if only_three:
            for point in threes:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(THREE, model_train_matrix[x][y])

            for point in block_fours:
                x = point // self.size
                y = point % self.size
                model_train_matrix[x][y] = max(BLOCK_FOUR, model_train_matrix[x][y])
            return set(list(block_fours) + list(threes)), model_train_matrix

        block_threes = points[shapes['BLOCK_THREE']]
        two_twos = points[shapes['TWO_TWO']]
        twos = points[shapes['TWO']]

        for point in block_threes:
            x = point // self.size
            y = point % self.size
            model_train_matrix[x][y] = max(BLOCK_THREE, model_train_matrix[x][y])

        for point in two_twos:
            x = point // self.size
            y = point % self.size
            model_train_matrix[x][y] = max(TWO_TWO, model_train_matrix[x][y])

        for point in twos:
            x = point // self.size
            y = point % self.size
            model_train_matrix[x][y] = max(TWO, model_train_matrix[x][y])

        for point in block_fours:
            x = point // self.size
            y = point % self.size
            model_train_matrix[x][y] = max(BLOCK_FOUR, model_train_matrix[x][y])

        for point in threes:
            x = point // self.size
            y = point % self.size
            model_train_matrix[x][y] = max(THREE, model_train_matrix[x][y])

        mid = list(block_fours) + list(threes) + list(block_threes) + list(two_twos) + list(twos)
        res = set(mid[:5])
        for i in range(len(model_train_matrix)):
            for j in range(len(model_train_matrix)):
                if (i * len(model_train_matrix) + j) not in res:
                    model_train_matrix[i][j] = 0
        return res, model_train_matrix
