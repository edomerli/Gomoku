from __future__ import absolute_import, division, print_function
from math import sqrt, log
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

# NOTE: deterministic_test() requires BUDGET = 1000
# You can try higher or lower values to see how the AI's strength changes
BUDGET = 1000

class AI:
    # NOTE: modifying this block is not recommended because it affects the random number sequences
    def __init__(self, state):
        self.simulator = Game()
        self.simulator.reset(*state) #using * to unpack the state tuple
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
        _, action, action_win_rates = self.best_child(self.root, 0)

        return action, action_win_rates

    def select(self, node):

        # Select a child node within the currently explored tree
        # NOTE: deterministic_test() requires using c=1 for best_child()

        while node.is_terminal == False:
            # case 1: node is not fully expanded
            if len(node.untried_actions) != 0:
                break
            # case 2: node is fully expanded
            else:
                node = self.best_child(node, c=1)[0]

        return node

    def expand(self, node):

        # Add a new child node from an untried action and return this new node

        # node could be a terminal node
        if node.is_terminal:
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


        return best_child_node, best_action, action_ucb_table

    def backpropagate(self, node, result):

        # Backpropagate the information about winner

        while (node is not None):
            # IMPORTANT: each node should store the number of wins of its **parent** node's player, when choosing the action towards node as a child
            node.num_visits += 1
            if node.parent is not None:
                node.num_wins += result[node.parent.state[0]]
            node = node.parent



    def rollout(self, node):

        # Rollout / Simulation (called DefaultPolicy in the slides)

        self.simulator.reset(*node.state)

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
