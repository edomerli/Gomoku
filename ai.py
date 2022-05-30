from __future__ import absolute_import, division, print_function
from math import sqrt, log, inf
from game import Game, WHITE, BLACK, EMPTY
import copy
import time
import random

class Node:
    # NOTE: modifying this block is not recommended
    def __init__(self, state, actions, parent=None):
        self.state = (state[0], copy.deepcopy(state[1]))    # state is of the form (player, grid)
        self.num_wins = 0 #number of wins at the node
        self.num_visits = 0 #number of visits of the node
        self.parent = parent #parent node of the current node
        self.children = [] #store actions and children nodes in the tree as (action, node) tuples
        self.untried_actions = copy.deepcopy(actions) #store actions that have not been tried
        simulator = Game(*state)
        self.is_terminal = simulator.game_over

        # CONTEST ADDITION: minimax
        self.is_minimax_terminal = False
        self.minimax_winner = None

# NOTE: deterministic_test() requires BUDGET = 1000
# You can try higher or lower values to see how the AI's strength changes
BUDGET = 6000


# CONTEST ADDITION: caching
saved_root = None

class AI:
    # NOTE: modifying this block is not recommended because it affects the random number sequences
    def __init__(self, state):
        self.simulator = Game()
        self.simulator.reset(*state) #using * to unpack the state tuple

        # CONTEST ADDITION: caching
        # When retrieving root, the best_child (i.e. opponent, BLACK player) has been saved
        # We want to get the child of that best_child (i.e. a WHITE player, A.I) that is in the same state of the game
        # i.e. travel through the action that the opponent has taken
        if saved_root is not None:
            for action, child in saved_root.children:
                if child.state[1] == state[1]:
                    self.root = child

        else:
            self.root = Node(state, self.simulator.get_actions())

    def mcts_search(self):

        # Implement the MCTS Loop

        iters = 0
        action_win_rates = {} #store the table of actions and their ucb values

        while(iters < BUDGET):
            if ((iters + 1) % 100 == 0):
                # NOTE: if your terminal driver doesn't support carriage returns you can use:
                # print("{}/{}".format(iters + 1, BUDGET))
                print("\riters/budget: {}/{}".format(iters + 1, BUDGET), end="")

            # SELECT
            selected_node = self.select(self.root)
            # EXPAND
            expanded_node = self.expand(selected_node)
            # SIMULATE
            result = self.rollout(expanded_node)
            # BACKPROPAGATE
            self.backpropagate(expanded_node, result)

            iters += 1
        print()

        # Note: Return the best action, and the table of actions and their win values
        #   For that we simply need to use best_child and set c=0 as return values
        best_child, action, action_win_rates = self.best_child(self.root, 0)

        # CONTEST ADDITION: caching
        # Since we will take the action that leeds to best_child, save best_child as root for next MCTS
        global saved_root
        saved_root = copy.deepcopy(best_child)

        return action, action_win_rates

    def select(self, node):

        # Select a child node within the currently explored tree
        # NOTE: deterministic_test() requires using c=1 for best_child()

        # CONTEST ADDITION: minimax
        while node.is_terminal == False and node.is_minimax_terminal == False:
            # case 1: node is not fully expanded
            if len(node.untried_actions) != 0:
                break
            # case 2: node is fully expanded
            else:
                node = self.best_child(node, c=1)[0]

        return node

    def expand(self, node):

        # Add a new child node from an untried action and return this new node

        # selected node could be a terminal node
        # CONTEST ADDITION: minimax
        if node.is_terminal or node.is_minimax_terminal:
            return node

        # NOTE: passing the deterministic_test() requires popping an action like this
        action = node.untried_actions.pop(0)

        self.simulator.reset(*(node.state))
        self.simulator.place(*action)
        new_state = self.simulator.state()

        child_node = Node(new_state, self.simulator.get_actions(), parent=node) # choose a child node to grow the search tree

        # Update the parent node, i.e. add child_node to the explored tree
        node.children.append((action, child_node))

        return child_node

    def best_child(self, node, c=1):

        # Determine the best child and action by applying the UCB formula

        best_child_node = None # to store the child node with best UCB
        best_action = None # to store the action that leads to the best child
        action_ucb_table = {} # to store the UCB values of each child node (for testing)

        best_ucb_value = -1
        log_parent_visits = log(node.num_visits)
        # NOTE: deterministic_test() requires iterating in this order
        for child in node.children:
            # NOTE: deterministic_test() requires, in the case of a tie, choosing the FIRST action with
            # the maximum upper confidence bound
            action = child[0]
            child_node = child[1]

            ucb_value = child_node.num_wins / child_node.num_visits + c * sqrt((2 * log_parent_visits) / child_node.num_visits)
            if ucb_value > best_ucb_value:
                best_ucb_value = ucb_value

                best_child_node = child_node
                best_action = action

            if c == 0: # i.e. we are returning the best action to take at the root node
                action_ucb_table[action] = ucb_value

        # CONTEST ADDITION: minimax
        if best_ucb_value == -1:
            # It means that it's impossible for this node to ever win, i.e. child_node.num_wins = -inf for each child_node
            node.is_minimax_terminal = True
            node.minimax_winner = BLACK if node.state[0] == WHITE else WHITE
            # The parent should come here since it would win (i.e. this node would lose)
            node.num_wins = inf

            if node.parent is not None:
                # The parent's parent shouldn't instead (otherwise it would lose because the parent would come here)!
                node.parent.num_wins = -inf
                # this way the parent is a minimax terminal itself
                node.parent.is_minimax_terminal = True
                node.parent.minimax_winner = node.parent.state[0]

            return node.children[0][1], node.children[0][0], {} # return a random action and children
        return best_child_node, best_action, action_ucb_table

    def backpropagate(self, node, result):

        # Backpropagate the information about winner
        if node.is_terminal:    # truly terminal node (not minimax)
            # update the minimax information
            node.num_wins = inf
            # don't change the info of node to minimax_terminal because it's already terminal (stronger property)

            # the parent becomes a minimax_terminal (coming to the parent from the grandpa would leed the grandpa to lose)
            node.parent.num_wins = -inf
            node.parent.is_minimax_terminal = True
            node.parent.minimax_winner = node.parent.state[0]


        while (node is not None):
            # IMPORTANT: each node should store the number of wins of its **parent** node's player, when choosing the action towards node as a child
            node.num_visits += 1
            if node.parent is not None:
                node.num_wins += result[node.parent.state[0]]
            node = node.parent



    def rollout(self, node):

        # Rollout / Simulation (called DefaultPolicy in the slides)

        self.simulator.reset(*node.state)

        # CONTEST ADDITION: minimax
        # If you have to run a simulation from a simulation from a minimax terminal node:
        # don't and just return the reward s.t. the minimax_winner has won
        reward = {}
        if node.is_minimax_terminal:
            if node.minimax_winner == BLACK:
                reward[BLACK] = 1
                reward[WHITE] = 0
            elif node.minimax_winner == WHITE:
                reward[BLACK] = 0
                reward[WHITE] = 1
            return reward

        # Simulate a game randomly
        # NOTE: deterministic_test() requires that you select a random move using self.simulator.rand_move()
        while not self.simulator.game_over:
            self.simulator.place(*(self.simulator.rand_move()))

        # Determine reward indicator from result of rollout
        reward = {}
        if self.simulator.winner == BLACK:
            reward[BLACK] = 1
            reward[WHITE] = 0
        elif self.simulator.winner == WHITE:
            reward[BLACK] = 0
            reward[WHITE] = 1
        return reward

#     CONTEST ADDITION IDEA: restrict the action choices only to dangerous positions
#     i.e. where THE OPPONENT (would have to modify the code) can win in the next 2 moves if I don't defend
#     def restrict_actions(self):
#         actions = []
#         for i in range(self.simulator.min_r, self.simulator.max_r + 1):
#             for j in range(self.simulator.min_c, self.simulator.max_c - 3):
#                 diz = freq(self.simulator.grid[i][j:j+5])
#                 if (diz['b'] >= 3 or diz['w'] >= 3) and diz['.'] == 5 - max(diz['b'], diz['w']):
#                     for k in range(j, j+5):
#                         if self.simulator.grid[i][k] == '.':
#                             actions.append((i, k))
#         # vertical
#         for j in range(self.simulator.min_c, self.simulator.max_c + 1):
#             for i in range(self.simulator.min_r, self.simulator.max_r - 3):
#                 diz = {'b':0, 'w':0, '.':0}
#                 for k in range(i, i+5):
#                     diz[self.simulator.grid[k][j]] += 1
#                 if (diz['b'] >= 3 or diz['w'] >= 3) and diz['.'] == 5 - max(
#                         diz['b'], diz['w']):
#                     for k in range(i, i + 5):
#                         if self.simulator.grid[k][j] == '.':
#                             actions.append((k, j))
#         # diagonal
#         if len(actions) == 0:
#             return self.simulator.get_actions()
#         return actions

# def freq(lista):
#     all_freq = {'b':0, 'w':0, '.':0}

#     for i in lista:
#         if i in all_freq:
#             all_freq[i] += 1
#         else:
#             all_freq[i] = 1

#     return all_freq
