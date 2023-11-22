"""
FileName: main_worker.py
Author: Jiaxin Li
Create Date: 2023/11/21
Description: The implement of  Gumbel MCST 
Edit History:
Debug: the dim of output: probs
"""

import numpy as np
import copy
import time 

from config.options import *
import sys
from config.utils import *


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def _sigma_mano(y ,Nb):
    return (50 + Nb) * 1.0 * y 


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._v = 0
        self._p = prior_p



    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
            

    def select(self, v_pi):
        """Select action among children that gives maximum 
        (pi'(a) - N(a) \ (1 + \sum_b N(b)))
        Return: A tuple of (action, next_node)
        """
        # if opts.split == "train":
        #     v_pi = v_pi.detach().numpy()
        # print(v_pi)
        

        

        max_N_b = np.max(np.array( [act_node[1]._n_visits   for act_node in   self._children.items()]))

        if opts.split == "train":
            pi_ = softmax( np.array( [ act_node[1].get_pi(v_pi,max_N_b)  for act_node in   self._children.items() ])).reshape(len(list(self._children.items())) ,-1)
        else:
            pi_ = softmax( np.array( [ act_node[1].get_pi(v_pi,max_N_b) for act_node in   self._children.items() ])).reshape(len(list(self._children.items())) ,-1)
        # print(pi_.shape)
        

        N_a = np.array( [ act_node[1]._n_visits / (1 + self._n_visits)   for act_node in   self._children.items() ]).reshape(pi_.shape[0],-1)
        # print(N_a.shape)    

        max_index=  np.argmax(pi_ - N_a)
        # print((pi_ - N_a).shape)
        

        
        return  list(self._children.items())[max_index]


    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        if opts.split == "train":
            self._Q = self._Q +  (1.0*(leaf_value  - self._Q ) / self._n_visits)
         
            
        else: 
            self._Q += (1.0*(leaf_value - self._Q) / self._n_visits)

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_pi(self,v_pi,max_N_b):
        if self._n_visits == 0:
            Q_completed = v_pi
        else:
            Q_completed = self._Q
        
        return  self._p + _sigma_mano(Q_completed,max_N_b)


    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class Gumbel_MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout




    def Gumbel_playout(self, child_node, child_state):
        """Run a single playout from the child of the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        This mothod of select is a non-root selet.
        """
        node = child_node
        state = child_state
    
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.

            action, node = node.select(node._v)
        
            state.do_move(action)



        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
    
        leaf_value = leaf_value.detach().numpy()[0][0]

        node._v = leaf_value
    


        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)
   

    def top_k(self,x, k):
        print("x",x.shape)
        print("k ", k)

        return np.argpartition(x, k)[..., -k:]

    def sample_k(self,logits, k):
        u = np.random.uniform(size=np.shape(logits))
        z = -np.log(-np.log(u))

  
        
        return self.top_k(logits + z, k),z


    def get_move_probs(self, state, temp=1e-3,m_action = 16):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        # 这里需要修改：1
        # logits 暂定为 p

        n = self._n_playout
        m = m_action

        # 对根节点进行拓展
        act_probs, leaf_value = self._policy(state)
        act_probs =  list(act_probs)

        leaf_value = leaf_value.detach().numpy()[0][0]
        
        # print(list(act_probs))
        porbs = [prob  for act,prob in (act_probs)]
        self._root.expand(act_probs)


        # 先进行Gumbel 分布采样，不重复的采样前m个动作，对应选择公式 logits + g
        A_topm ,g = self.sample_k(porbs , m)
        
        # 获得state选取每个action后对应的状态，保存到一个列表中
        root_childs = list(self._root._children.items())
 

        child_state_m = []
        for i in range(m):
            state_copy = copy.deepcopy(state)
            action,node = root_childs[A_topm[i]]
            state_copy.do_move(action)
            child_state_m.append(state_copy)

           
        # 每轮对选择的动作进行的仿真次数
        N = int( n /( np.log(m) * m ))

        # 进行sequential halving with Gumbel 
        while m >= 1:
        
            # 对每个选择的动作进行仿真
            for i in range(m):
                action_state = child_state_m[i]
            
                action,node = root_childs[A_topm[i]]
            
                for j in range(N):
                    action_state_copy = copy.deepcopy(action_state)
        
                    # 对选择动作进行仿真: 即找到这个子树的叶节点，然后再网络中预测v，然后往上回溯的过程
                    self.Gumbel_playout(node, action_state_copy)

            # 每轮不重复采样的动作个数减半
            m = m //2

            # 不是最后一轮,单轮仿真次数加倍
            if(m != 1):
                n = n - N
                N *= 2
            # 当最后一轮时,只有一个动作,把所有仿真次数用完
            else:
                N = n
            
            # 进行新的一轮不重复采样, 采样在之前的动作前一半的动作, 对应公式 g + logits + \sigma( \hat{q} )
            # print([action_node[1]._Q for action_node in self._root._children.items()  ])
            
       
            q_hat = np.array([action_node[1]._Q for action_node in self._root._children.items()  ])
            

            assert(np.sum(q_hat[A_topm] == 0) == 0  )

            A_index = self.top_k( np.array(porbs)[A_topm] +  np.array(g)[A_topm]  +  q_hat[A_topm]  , m)
            A_topm = np.array(A_topm)[A_index]
            child_state_m = np.array(child_state_m)[A_index]
            
        
        # 最后返回对应的决策函数, 即 pi' = softmax(logits + sigma(completed Q))

        max_N_b = np.max(np.array( [act_node[1]._n_visits   for act_node in   self._root._children.items()]  ))

        final_act_probs=    softmax( np.array( [ act_node[1].get_pi(leaf_value, max_N_b)   for act_node in   self._root._children.items() ]))
        action =  ( np.array( [ act_node[0]   for act_node in   self._root._children.items() ]))

        return   np.array(list(self._root._children.items()))[A_topm][0][0], action,  final_act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class Gumbel_MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0,m_action = 16):
        self.mcts = Gumbel_MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.m_action = m_action


    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    
    def get_action(self, board, temp=1e-3, return_prob=0,return_time = False):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        
        
        
        if len(sensible_moves) > 0:
            start = time.time()
            # 在搜索树中利用sequential halving with Gumbel 来进行动作选择 并且返回对应的决策函数
            move, acts, probs = self.mcts.get_move_probs(board, temp,self.m_action)
    

            # 重置搜索树
            self.mcts.update_with_move(-1)
            move_probs[list(acts)] = probs
            if return_time:
                print("[OBSERVER] get a move need", time.time() - start)
            
        

            if return_prob:
                
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
