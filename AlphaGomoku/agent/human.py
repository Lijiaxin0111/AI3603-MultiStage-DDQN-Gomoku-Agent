from .agent import Agent
from .mcts import coordinate2index,index2coordinate
import numpy as np


class HumanAgent(Agent):
    def __init__(self, renderer, color, board_size):
        self._renderer = renderer
        self._color = color
        self._board_size = board_size

    def set_renderer(self, renderer):
        self._renderer = renderer

    def play(self, obs, action, stone_num, *args):
        x, y = self._renderer.ask_for_click()
        ind = coordinate2index((x, y), self._board_size)
        pi = np.zeros(self._board_size * self._board_size)
        pi[ind] = 1
        return (x, y), pi, None, None
    
class TerminalAgent(Agent):
    def __init__(self, renderer, color, board_size):
        self._renderer = renderer
        self._color = color
        self._board_size = board_size

    def set_renderer(self, renderer):
        self._renderer = renderer

    def play(self, obs, action, stone_num, *args):
   
        act_ind =  int(input("Input your move: "))
        act_cor = index2coordinate(act_ind, self._board_size)
        # print("x y :",(x,y))
  
        pi = np.zeros(self._board_size * self._board_size)
        pi[act_ind] = 1
        return act_cor, pi, None, None 
