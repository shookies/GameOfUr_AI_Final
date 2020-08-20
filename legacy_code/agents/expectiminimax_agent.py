from agents.Agents import Agent
from agents.util import raiseNotDefined
import numpy as np


# CONSTANTS

MAX = 1
MIN = 0
CENTRAL_POS = 1

# hyper parameters for heuristic
DOUBLE_REWARD = 16
CENTRAL_REWARD = 10
CENTRAL_PENALTY = -10
DANGER_PENALTY = -5
EATING_POTENTIAL_REWARD = 5

# Weights
MY_PIECES_WEIGHT = 1
OPPONENT_PIECES_WEIGHT = -2
CENTRAL_CONTROL_WEIGHT = 2
DANGER_WEIGHT = -5
POTENTIAL_EATING_WEIGHT = 2
MY_PIECES_OFF_PENALTY = 0
MY_PIECES_OUT_REWARD = 100
OPPONENT_OFF_REWARD = 100
OPPONENT_OUT_PENALTY = -2

class ExpectiminimaxAgent(Agent):
    """
    This agent searches through a tree up to a certain depth for the next move
    the expectimax algorithm returns the move whose weighted sum is the best.
    """

    def __init__(self,  index, opponent_index, depth=3):
        """
        intializer for the agent
        :param depth: how deep should the search last
        """
        self.index = index
        self.opponent_index = opponent_index
        self.depth = depth
        # probability distribution for each roll outcome
        self.probability = {0: 1/16, 1: 4/16, 2: 6/16, 3: 4/16, 4: 1/16}
        ranking_white = {(0, 4): 0, (0, 3): 1, (0, 2): 2, (0, 1): 3, (0, 0): 4,
                              (1, 0): 5, (1, 1): 6, (1, 2): 7, (1, 3): 8, (1, 4): 9, (1, 5): 10, (1, 6): 11, (1, 7): 12,
                              (0, 7): 13, (0, 6): 14}
        ranking_black = {(2, 4): 0, (2, 3): 1, (2, 2): 2, (2, 1): 3, (2, 0): 4,
                              (1, 0): 5, (1, 1): 6, (1, 2): 7, (1, 3): 8, (1, 4): 9, (1, 5): 10, (1, 6): 11, (1, 7): 12,
                              (2, 7): 13, (2, 6): 14}
        self.ranking = ranking_white if index == 'W' else ranking_black
        self.reversed_ranking = {value: key for (key, value) in self.ranking.items()}

        self.opponent_ranking = ranking_white if index == 'B' else ranking_white
        self.reversed_opponent_ranking = {value: key for (key, value) in self.opponent_ranking.items()}

        rosette_positions_white = [(0, 0), (1, 3), (0, 6)]
        rosette_positions_black = [(2, 0), (1, 3), (2, 6)]
        self.rosette_positions = rosette_positions_white if index == 'W' else rosette_positions_black



    def get_action(self, state, reward):
        """
        returns an action given the state
        :param state: the board state
        :param reward: None - not relevant to this agent
        :return: a legal action Tuple(int,int)
        """
        from engine.game_engine import Board
        roll = state.roll_dice()
        roll_sum = sum(roll)
        legal_actions = state.get_legal_moves()
        if roll_sum == 0 or not legal_actions:
            return None
        action_scores = []
        successors = Board.get_successors(state.get_current_board(), self.index)
        successors_of_roll = successors[roll_sum]
        for successor_state, cur_turn in successors_of_roll:
            #TODO is order preversed between the legal actions and the successors of that roll?!?
            action_scores.append(self.expectiminimax(cur_depth=0, state=successor_state, chance=True, roll=None, cur_turn=cur_turn))
        best_action_index = np.argmax(action_scores)
        return legal_actions[best_action_index]

    def expectiminimax(self, cur_depth, state, chance=False, roll=None, cur_turn=None):
        """
        returns expected minimax value of a two player stochastic game
        :param cur_depth: the current depth
        :param state: the current board state (must consist of whose turn it is)
        :param chance: bool whether or not we're in a chance node.
        :param cur_turn: indicates whose turn it is for the given state
        :return:
        """

        from engine.game_engine import Board
        if cur_depth == self.depth or is_game_over(state):
            return self.evaluate(state)
        if chance:
            alpha = 0
            for cur_roll in range(5):  # 0 to 4
                alpha += self.probability[cur_roll] * self.expectiminimax(cur_depth, state, chance=False, roll=cur_roll, cur_turn=cur_turn)
        elif cur_turn == self.index:
            # set max node
            alpha = -np.inf
            if roll is not None:
                successors = Board.get_successors(state, cur_turn)
                successors_of_roll = successors[roll]
                # roll landed but there are no legal moves, we act as if we got 0
                if not successors_of_roll:
                    successors_of_roll = successors[0]
                for child_state, cur_turn in successors_of_roll:
                    children_score = self.expectiminimax(cur_depth, child_state, chance=True, cur_turn=cur_turn)
                    alpha = np.max([alpha, children_score])
        else:
            # min node
            alpha = np.inf
            if roll is not None:
                successors = Board.get_successors(state, cur_turn)
                successors_of_roll = successors[roll]
                # roll landed but there are no legal moves, we act as if we got 0
                if not successors_of_roll:
                    successors_of_roll = successors[0]
                for child_state, cur_turn in successors_of_roll:
                    children_score = self.expectiminimax(cur_depth+1, child_state, chance=True, cur_turn=cur_turn)
                    alpha = np.min([alpha, children_score])
        return alpha


    def evaluate(self, state):
        """
        given a state evaluate gives a value to the state given the calculated heuristic
        as proposed here
        :param state: the board state
        :return: a float value
        """
        # a couple of options:

        # how many of my pieces are on the board
        board = state
        my_pieces_counter = 0
        opponent_pieces_counter = 0
        for row in board:
            for tile in row:
                if tile == self.index:
                    my_pieces_counter += 1
                if tile == self.opponent_index:
                    opponent_pieces_counter += 1

        # potential for double turn
        double_potential = 0
        for roll in range(1, 5):
            for rosette in self.rosette_positions:
                board_pos_x, board_pos_y = self.reversed_ranking[self.ranking[rosette] - roll]
                if board[board_pos_x][board_pos_y] == self.index:
                    double_potential += self.probability[roll] * DOUBLE_REWARD

        # control of central rossete
        central_control_reward = 0
        board_pos = self.rosette_positions[CENTRAL_POS]
        if board[board_pos[0]][board_pos[1]] == self.index:
            central_control_reward = CENTRAL_REWARD
        elif board[board_pos[0]][board_pos[1]] == self.opponent_index:
            central_control_reward = CENTRAL_PENALTY

        my_pieces_in_danger_score = 0
        potential_eating_score = 0
        # how many of my pieces are at a real risk of being eaten next turn
        # my pieces can only be eaten in the main row, use opponent ranking to see the risk.
        middle_row = board[CENTRAL_POS]
        for index, tile in enumerate(middle_row):
            if tile == self.index:
                # check previous 1 to 4 tiles according to opponent ranking and see if an opponent piece is there
                for roll in range(1,5):
                    board_pos_x, board_pos_y = self.reversed_opponent_ranking[self.opponent_ranking[(1, index)] - roll]
                    if board[board_pos_x][board_pos_y] == self.opponent_index:
                        my_pieces_in_danger_score += self.probability[roll] * DANGER_PENALTY

            # how many pieces can I potentially eat?
            elif tile == self.opponent_index:
                for roll in range(1,5):
                    board_pos_x, board_pos_y = self.reversed_ranking[self.ranking[(1, index)] - roll]
                    if board[board_pos_x][board_pos_y] == self.index:
                        potential_eating_score += self.probability[roll] * EATING_POTENTIAL_REWARD

        # how many pieces do me and my opponent have off the board and out the game
        my_pieces_off_board, my_pieces_out_game = get_off_and_out_pieces(board, self.index)
        opponent_pieces_off_board, opponent_pieces_out_game = get_off_and_out_pieces(board, self.opponent_index)

        # use linear combination of these parameters as the heuristic
        return my_pieces_counter*MY_PIECES_WEIGHT + opponent_pieces_counter * OPPONENT_PIECES_WEIGHT + \
               central_control_reward * CENTRAL_CONTROL_WEIGHT + my_pieces_in_danger_score * DANGER_WEIGHT + \
            potential_eating_score * POTENTIAL_EATING_WEIGHT + my_pieces_off_board * MY_PIECES_OFF_PENALTY \
    + my_pieces_out_game * MY_PIECES_OUT_REWARD + opponent_pieces_off_board * OPPONENT_OFF_REWARD  + \
    opponent_pieces_out_game * OPPONENT_OUT_PENALTY


def get_off_and_out_pieces(board, index):
    """
    helper function which returns how many pieces I have off and out the game depending on index
    :param board: board state
    :param index: indicating which player we're asking for
    :return: Tuple(int, int)
    """
    if index == 'W':
        return (board[0][4], board[0][5])

    else:
        return (board[2][4], board[2][5])

def is_game_over(board):
    """
    checks if the game is over (one of the player have moved 7 pieces to the exit)
    :return: 'W' if white player won, 'B' if black player won, None o.w
    :rtype: str of None
    """
    if board[0][5] == 7:
        return 'W'
    if board[2][5] == 7:
        return 'B'
    return None

