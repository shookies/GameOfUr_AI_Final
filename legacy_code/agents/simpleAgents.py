from agents.Agents import Agent
from agents.util import raiseNotDefined
import random


class RandomAgent(Agent):

    def __init__(self, index):
        """
        initializer for the random agent
        """
        self.index = index

    def get_action(self, state, reward):
        """
        returns a random action
        :param state: the board state
        :param reward: None - not relevant for this agnet
        :return:
        """
        dice = state.roll_dice()
        if sum(dice) == 0:
            return None
        actions = self.get_legal_actions(state)
        if not actions:
            return None
        random_move = random.choice(actions)
        return random_move

    def get_legal_actions(self, state):
        """
        returns list of legal actions
        :param state: the board state
        :return:
        """
        return state.get_legal_moves()


class GreedyAgent(Agent):
    """
    This class describes a greedy agent, given a move it will make the move which
    moves the piece closest to going out of the board.
    """

    def __init__(self, index):
        """
        initialize the greedy agent
        :param actionFn: getLegalAction function
        :param index 'W' or 'B' for white or black
        """
        self.index = index
        self.ranking_white = {(0, 4): 0, (0, 3): 1, (0, 2): 2, (0, 1): 3, (0, 0): 4,
                              (1, 0): 5, (1, 1): 6, (1, 2): 7, (1,3): 8, (1,4): 9, (1,5): 10, (1,6): 11, (1, 7): 12,
                              (0, 7): 13, (0, 6): 14}
        self.ranking_black = {(2,4): 0, (2, 3): 1, (2, 2): 2, (2, 1): 3, (2, 0): 4,
                              (1, 0): 5, (1, 1): 6, (1, 2): 7, (1,3): 8, (1,4): 9, (1,5): 10, (1,6): 11, (1, 7): 12,
                              (2, 7): 13, (2, 6): 14}
        self.ranking = self.ranking_white if index == 'W' else self.ranking_black

    def get_action(self, state, reward):
        """
        this method receives a state and returns a move which will take a piece furthest in the board.
        :param state: state of the board
        :param reward: not used in this agent
        :return:
        """
        dice = state.roll_dice()
        if sum(dice) == 0:
            return None
        actions = self.get_legal_actions(state)
        if not actions:
            return None
        best_move = self.get_best_move(actions)
        return best_move

    def get_best_move(self, actions):
        """
        returns the best move for the greedy agent, which is the move which
        moves a piece furthest along the board
        :param state: board state
        :param actions: list of all legal actions that can be made given the state
        these are the positions from which we can move from
        """
        best_action_index = -1
        best_action = None
        for action in actions:
            if action not in self.ranking:
                # TODO this is here for debugging purposes only and should be deleted
                raise Exception("Illegal action trying to take place")
            else:
                if self.ranking[action] > best_action_index:
                    best_action_index = self.ranking[action]
                    best_action = action
        return best_action

    def get_legal_actions(self, state):
        """
        returns list of legal actions
        :return:
        """
        return state.get_legal_moves()

