"""
FileName: board_cut.py
Author: Jiaxin Li
Create Date: 2023/11/22
Description: The implement of some Gomoku_trick
Edit History:
"""
from scipy.ndimage import binary_dilation
import numpy as np

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def board_cut(state,procs, bound = 2):
    # board_width =  state.shape[1]
    # board_height = state.shape[0]
    # board_zero = np.zeros(board_width ,board_height)
   
    state = state.current_state()



    board_state = (state[0] + state[1]) > 0
    print("board_ state:",board_state.reshape(8,8))
    if np.sum(board_state) == 0:
        
        return procs
    # print(board_state)
    result = binary_dilation(board_state, structure=np.ones((2 * bound  + 1, 2 * bound  + 1)))

    index = result.reshape(len(procs))

    init_procs = procs

    procs = procs * index

    print("after procs",procs.reshape(8,8))
    if np.sum( procs > 0) == 0:
        return init_procs

    procs[procs  > 0] = softmax(  procs[procs>0  ])

    print(sum( procs))


    print("after",procs)
    return procs

if __name__ == "__main__":
    board_cut(np.array([[1]]),[1],1,1)
    print("ss")
        
    