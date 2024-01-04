from .cache import Cache
from .eval import FIVE
# Checked


MAX = 1000000000
cache_hits = {
    "search": 0,
    "total": 0,
    "hit": 0
}

onlyThreeThreshold = 6
cache = Cache()


def factory(onlyThree=False, onlyFour=False):

    def helper(board, role, depth, cDepth=0, path=(), alpha=-MAX, beta=MAX):
        cache_hits["search"] += 1
        if cDepth >= depth or board.isGameOver():
            return [board.evaluate(role), None, (path)]
        hash = board.hash()
        prev = cache.get(hash)
        if prev and prev["role"] == role:
            if (
                (abs(prev["value"]) >= FIVE or prev["depth"] >= depth - cDepth)
                and prev["onlyThree"] == onlyThree
                and prev["onlyFour"] == onlyFour
            ):
                cache_hits["hit"] += 1
                return [prev["value"], prev["move"], path + prev["path"]]
        value = -MAX
        move = None
        bestPath = path  # Copy the current path
        bestDepth = 0
        # points = board.getValuableMoves(role, cDepth, onlyThree or cDepth > onlyThreeThreshold, onlyFour)
        points = board.getValuableMoves(role, cDepth, onlyThree or cDepth > onlyThreeThreshold, onlyFour)
        if cDepth == 0:
            print('points:', points)
        if not len(points):
            return [board.evaluate(role), None, path]
        for d in range(cDepth + 1, depth + 1):
            # 迭代加深过程中只找己方能赢的解，因此只搜索偶数层即可
            if d % 2 != 0:
                continue
            breakAll = False
            for point in points:
                board.put(point[0], point[1], role)
                newPath = tuple(list(path) + [point])  # Add current move to path
                currentValue, currentMove, currentPath = helper(board, -role, d, cDepth + 1, tuple(newPath) , -beta, -alpha)
                currentValue = -currentValue
                board.undo()
                ## 迭代加深的过程中，除了能赢的棋，其他都不要
                ## 原因是：除了必胜的，其他评估不准。比如必输的棋，由于走的步数偏少，也会变成没有输，比如5
                ### 步之后输了，但是1步肯定不会输，这时候1步的分数是不准确的，显然不能选择。
                if currentValue >= FIVE or d == depth:
                    # 必输的棋，也要挣扎一下，选择最长的路径
                    if (
                        currentValue > value
                        or (currentValue <= -FIVE and value <= -FIVE and len(currentPath) > bestDepth)
                    ):
                        value = currentValue
                        move = point
                        bestPath = currentPath
                        bestDepth = len(currentPath)
                alpha = max(alpha, value)
                if alpha >= FIVE:
                    breakAll = True
                    break
                if alpha >= beta:
                    break
            if breakAll:
                break
        if (cDepth < onlyThreeThreshold or onlyThree or onlyFour) and (not prev or prev["depth"] < depth - cDepth):
            cache_hits["total"] += 1
            cache.put(hash, {
                "depth": depth - cDepth,
                "value": value,
                "move": move,
                "role": role,
                "path": bestPath[cDepth:],
                "onlyThree": onlyThree,
                "onlyFour": onlyFour,
            })
        return [value, move, bestPath]
    return helper


_minmax = factory()
vct = factory(True)
vcf = factory(False, True)


def minmax(board, role, depth=4, enableVCT=False):

    if enableVCT:
        vctDepth = depth + 8
        value, move, bestPath = vct(board, role, vctDepth)
        if value >= FIVE:
            return [value, move, bestPath]
        value, move, bestPath = _minmax(board, role, depth)
        '''
        // 假设对方有杀棋，先按自己的思路走，走完之后看对方是不是还有杀棋
        // 如果对方没有了，那么就说明走的是对的
        // 如果对方还是有，那么要对比对方的杀棋路径和自己没有走棋时的长短
        // 如果走了棋之后路径变长了，说明走的是对的
        // 如果走了棋之后，对方杀棋路径长度没变，甚至更短，说明走错了，此时就优先封堵对方
        '''
        board.put(move[0], move[1], role)
        value2, move2, bestPath2 = vct(board.reverse(), role, vctDepth)
        board.undo()
        if value < FIVE and value2 == FIVE and len(bestPath2) > len(bestPath):
            value3, move3, bestPath3 = vct(board.reverse(), role, vctDepth)
            if len(bestPath2) <= len(bestPath3):
                return [value, move2, bestPath2] # value2 是被挡住的，所以这里还是用value
        return [value, move, bestPath]
    else:
        return _minmax(board, role, depth)