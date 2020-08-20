"""
This file contains code that is essentially the game engine.
It will be used both by the algorithms and by the GUI
rules specified in this file are the basic rules (7 players, short path).
"""

import random
import copy
import time
import simpleAgents as simpleAgents
import expectiminimax_agent as expectiminimax_agent
import Agents as Agents
import gui
import argparse



EMPTY_SQUARE = '_'
BOARD = 0
PLAYER = 1
MIN_MOVE_DELAY = 0.05

MAIN_MENU_HELP = 'If given, the game initiates WITHOUT the main menu, giving the ability to select players while ignoring command line arguments'
BLACK_PLAYER_HELP = 'Determines the black player. 1 for Human, 2 for Expectimax, 3 for Deep-Q, 4 for Random Playing Agent, 5 for Greedy, 6 for alternative DeepQ'
WHITE_PLAYER_HELP = 'Determines the white player. 1 for Human, 2 for Expectimax, 3 for Deep-Q, 4 for Random Playing Agent, 5 for Greedy, 6 for alternative DeepQ'
EXPECTIMAX_DEPTH_HELP = 'Determines Expectimax search depth (default == 1)'
NUM_OF_GAMES_HELP = 'Determines number of games to be played (Useful for showcasing agent play)'
DELAY_HELP = 'Determines the amount of delay between actions of each agent (default == 1). NOTE: depth of more than 1 will result in slow turns'
GAME_MODE_HELP = 'Decide upon game mode, a single game with the menu or multiple games with more control on the hyperparameters, default is a single game mode,' \
                 '0 for single mode, 1 for multiple games'
LOAD_PATH_HELP = 'Load weights from a path for a Q learning agent, default is our best trained agent that has been provided'
SAVE_PATH_HELP = 'Path where weights will be saved with a learning Q Agent'
LEARNING_HELP = 'bool whether or not the learning agent will learn. default is False 0 - False 1 - True'
GUI_HELP = '-gui 0 for no GUI and 1 to activate it, default is no GUI (this is for multiple games only)'

class Board:

    def __init__(self):
        """
        creates a board object
        """
        self.__board = self.__create_new_board()
        self.__prev_board = None
        # self.__score = 0
        # self.roll_dice()
        self.__current_dice_roll = None
        self.__turn = None
        self.__opponent = None
        self.__base_row_dict = {'W': 0, 'B': 2}
        self.__player_W = None
        self.__player_B = None
        self.__probabilities = {1: 4/16, 2: 6/16, 3: 4/16, 4: 1/16}

    def __create_new_board(self):
        """
        creates a new game board represented by a 3x8 matrix, where each entry is a square on the board:
        - squares (0,0) - (0,3): WHITE player start base (before main alley)
        - squares (0,6) - (0,7): WHITE player end base (after main alley)
        - square (0,4): WHITE player starting point (contains number of pieces yet to be on board) - 7
        - square (0,5): WHITE player finish point (contains number of pieces that are off the board) - 0

        (symmetry applied for the BLACK player, for row number 2)

        - squares (1,0) - (1,7): main alley contains list of pieces, i.e ['W','B'], ['B']
        :return: game board matrix
        :rtype: list of lists
        """
        board = [[EMPTY_SQUARE] * 8] + [[EMPTY_SQUARE] * 8] + [[EMPTY_SQUARE] * 8]
        board[0][4] = 7
        board[0][5] = 0
        board[2][4] = 7
        board[2][5] = 0
        self.__board = board
        return board

    def get_cur_turn(self):
        """
        :return: Current player's turn
        """
        return self.__turn

    def get_current_board(self):
        """
        :return: copy of the current board
        :rtype: list of lists
        """
        return copy.deepcopy(self.__board)

    def roll_dice(self):
        """
        4 binary dice roll
        :return: list of 4 ints (0 or 1)
        :rtype: list
        """
        dice_roll = random.choices(population=[0, 1], k=4)
        self.__current_dice_roll = dice_roll
        return dice_roll

    def get_base_row(self):
        """
        returns the base row of the current player. 0 if white, 2 if black
        :return: base row index
        :rtype: int
        """
        return self.__base_row_dict[self.__turn]

    def get_base_row_opponent(self):
        """
        returns the base row of the current opponent. 0 if white, 2 if black
        :return: base row index
        :rtype: int
        """
        return self.__base_row_dict[self.__opponent]

    def __find_next_pos(self, row, col):
        """
        find the next x,y position on the game board given some row, col position.
        uses the _current_dice_roll as the number of steps to use
        :param row: board row of current piece location
        :type row: int
        :param col: board column of current piece location
        :type col: int
        :return: None if resulting position is off board, (new_row, new_col) otherwise
        :rtype: None or (int,int)
        """
        base_row = self.get_base_row()
        part_0 = [(base_row, i) for i in range(4, -1, -1)]
        part_1 = [(1, i) for i in range(8)]
        part_2 = [(base_row, i) for i in range(7, 4, -1)]
        layout = part_0 + part_1 + part_2
        curr_position_index = layout.index((row, col))
        next_position_index = curr_position_index + sum(self.__current_dice_roll)
        return None if next_position_index >= len(layout) else layout[next_position_index]

    def __is_rosette(self, row, col):
        """
        checks if given square position is a rosette
        :param row: board row index
        :type row: int
        :param col: board column index
        :type col: int
        :return: True if is a rosette, False o.w
        :rtype: bool
        """
        return (row, col) in [(0, 0), (2, 0), (1, 3), (0, 6), (2, 6)]

    def __is_move_legal(self, row, col):
        """
        checks if a given piece can be moved
        :param row: row index on board
        :type row: int
        :param col: column index on board
        :type col: int
        :return: next position coordinates if move is legal, None o.w
        :rtype: (int,int) or None
        """
        base_row = self.get_base_row()
        if row == self.get_base_row_opponent():
            # trying to move a piece from other player's base
            return None
        if (row, col) == (base_row, 4) and (self.__board[base_row][4] == 0):
            # trying to put a piece on the board when there are no available pieces off board
            return None
        if self.__board[row][col] in [self.__opponent, EMPTY_SQUARE]:
            # trying to move from an empty square or square that contains an opponent's piece
            return None

        if (row, col) == (base_row, 5):
            # trying to move a piece from the exit
            return None

        next_position = self.__find_next_pos(row, col)

        if not next_position:
            # trying to move piece outside of the board (overshot the exit)
            return None
        if self.__board[next_position[0]][next_position[1]] == self.__turn:
            # trying to move piece to an occupied square by current's player piece
            return None
        if self.__board[next_position[0]][next_position[1]] == self.__opponent and next_position == (1, 3):
            # next position is center rosette and its occupied by the opponent
            return None

        return next_position

    def __move_piece(self, row, col):
        """
        moves piece from given square coordinates to the next position according to __current_dice_roll
        :param row: current row index on board
        :type row: int
        :param col: current column index on board
        :type col: int
        :return: next position coordinates move was successful, None o.w
        :rtype: (int,int) or None
        """
        base_row = self.get_base_row()
        next_position = self.__is_move_legal(row, col)
        if not next_position:
            return None

        if next_position == (base_row, 5):
            # moved a piece to the exit
            self.__board[base_row][5] += 1
            self.__board[row][col] = EMPTY_SQUARE
            return next_position

        if self.__opponent == self.__board[next_position[0]][next_position[1]]:
            # returning opponent's piece to his base
            self.__board[self.get_base_row_opponent()][4] += 1

        if (row, col) == (base_row, 4):
            self.__board[base_row][4] -= 1
        else:
            self.__board[row][col] = EMPTY_SQUARE
        self.__board[next_position[0]][next_position[1]] = self.__turn

        return next_position

    def position_if_moved(self, row, column):
        """
        return the future position resulting in moving a given piece, if the move is legal
        :param row: row index on board
        :type row: int
        :param col: column index on board
        :type col: int
        :return: next position coordinates if move is legal, None o.w
        :rtype: (int,int) or None
        """
        return self.__is_move_legal(row, column)

    def get_legal_moves(self):
        """
        returns list of legal coordinates that the current player can select to move
        :return: list of legal moves (coordinates)
        :rtype: list of (int,int)
        """
        legal_moves = []
        base_row = self.get_base_row()
        for column in range(8):
            if self.__is_move_legal(base_row, column):
                legal_moves.append((base_row, column))
            if self.__is_move_legal(1, column):
                legal_moves.append((1, column))
        return legal_moves

    @staticmethod
    def get_successors(state, player_color):
        """
        creates a dictionary of successor states of a given state for all the possible rolls of the given player color
        :param state: board representation as in Board.__board
        :type state: list of lists
        :param player_color: 'W' if the player that is about to play from the given state is White, 'B' o.w
        :type player_color: str
        :return: dictionary where the keys are the dice roll sum,
                 and the values are tuples of the possible successor states to that roll, and the next player to play
        :rtype: dict
        """
        move_dict = {0: [(state, 'B' if player_color == 'W' else 'W')], 1: [], 2: [], 3: [], 4: []}
        for dice_roll in range(1, 5):
            new_board = Board()
            new_board.__board = copy.deepcopy(state)
            new_board.__turn = player_color
            new_board.__opponent = 'W' if player_color == 'B' else 'B'
            new_board.__current_dice_roll = [1 if i <= dice_roll else 0 for i in range(1, 5)]
            legal_moves = new_board.get_legal_moves()
            for move in legal_moves:
                new_position = new_board.__move_piece(*move)
                if new_board.__is_rosette(*new_position):
                    move_dict[dice_roll].append((new_board.__board, new_board.__turn))
                else:
                    move_dict[dice_roll].append((new_board.__board, new_board.__opponent))

                new_board.__board = copy.deepcopy(state)

        return move_dict

    def __str__(self):
        """
        creates a string representation of the board
        like:
        ◬ ○ _ _ 7 0 ◬ _
        ○ _ _ ◬ _ _ _ _
        ● ● ● _ 7 0 ◬ _
        :return: board string
        :rtype: str
        """
        board_string = ""
        for row in range(3):
            for column in range(8):
                square = self.__board[row][column]
                str_to_add = str(square)
                if self.__is_rosette(row, column):
                    str_to_add = u"\u25EC"
                if square == 'W':
                    str_to_add = u"\u25CB"
                if square == 'B':
                    str_to_add = u"\u25CF"
                board_string += str_to_add + " "
            board_string += "\n"
        return board_string

    def current_player(self):
        """
        gets the current turn player
        :return: 'W' for white player, 'B' for black player
        :rtype: str
        """
        return self.__turn

    def get_current_dice(self):
        return self.__current_dice_roll

    def current_opponent(self):
        """
        gets the current turn opponent
        :return: 'W' for white opponent, 'B' for black opponent
        :rtype: str
        """
        return self.__opponent

    def is_game_over(self):
        """
        checks if the game is over (one of the player have moved 7 pieces to the exit)
        :return: 'W' if white player won, 'B' if black player won, None o.w
        :rtype: str of None
        """
        if self.__board[0][5] == 7:
            return 'W'
        if self.__board[2][5] == 7:
            return 'B'
        return None

    def __change_turn(self):
        """
        Changes turn
        :return: Current player
        """

        self.__turn, self.__opponent = self.__opponent, self.__turn
        return self.__player_W if self.__turn == 'W' else self.__player_B


    def run_game(self, black_player, white_player, GUI, delay=0):
        """
        Runs the main game loop.
        :param black_player:
        :type black_player: Agents.py class
        :param white_player:
        :type white_player: Agents.py class
        :param GUI:  The GUI object to draw the game graphics.
        :type GUI: GUI object
        :param delay: Delay between agent actions
        :type delay: int
        :return: The winner of the current game.
        :rtype: String
        """

        white_roll = 0
        black_roll = 0
        while white_roll == black_roll:
            white_roll = sum(self.roll_dice())
            black_roll = sum(self.roll_dice())
        self.__turn, self.__opponent = ('W', 'B') if white_roll > black_roll else ('B', 'W')
        self.__player_W = white_player
        self.__player_B = black_player
        current_player = self.__player_W if self.__turn == 'W' else self.__player_B

        winner = None
        while not winner:
            winner = self.is_game_over()
            if winner:
                break
            cur_p_str = 'White' if self.__turn == 'W' else 'Black'
            GUI.draw_board(self, self.__turn)
            if isinstance(current_player, Agents.Human):
                chosen_action = GUI.get_action(self)
            else:
                chosen_action = current_player.get_action(self, reward=None)
                GUI.update_msg(
                    cur_p_str + ' rolled ' + str(sum(self.__current_dice_roll)))
                time.sleep(MIN_MOVE_DELAY)
                time.sleep(delay)
            if not chosen_action:
                current_player = self.__change_turn()
                continue

            new_position = self.__move_piece(*chosen_action)

            while self.__is_rosette(*new_position):
                GUI.draw_board(self, self.__turn)
                time.sleep(MIN_MOVE_DELAY)
                if isinstance(current_player, Agents.Human):
                    chosen_action = GUI.get_action(self)
                else:
                    chosen_action = current_player.get_action(self,
                                                              reward=None)
                    GUI.update_msg(
                        cur_p_str + ' rolled ' + str(
                            sum(self.__current_dice_roll)))
                    time.sleep(MIN_MOVE_DELAY)
                    time.sleep(delay)
                if not chosen_action:
                    break
                new_position = self.__move_piece(*chosen_action)

            current_player = self.__change_turn()

        return winner
    def double_turn_potetntial(self):

        potential = 0
        base_row = self.get_base_row()
        part_0 = [(base_row, i) for i in range(4, -1, -1)]
        part_1 = [(1, i) for i in range(8)]
        part_2 = [(base_row, i) for i in range(7, 4, -1)]
        layout = part_0 + part_1 + part_2
        rosette_indices = [4,8,14]
        for i in range(len(layout)):
            if self.__board[layout[i][0]][layout[i][1]] == self.__turn:
                for index in rosette_indices:
                    if index - i in self.__probabilities:
                        potential += 16 * self.__probabilities[index - i]
        return potential

    def be_eaten_potential(self):
        potential = 0
        base_row = self.get_base_row()
        part_0 = [(base_row, i) for i in range(4, -1, -1)]
        part_1 = [(1, i) for i in range(8)]
        part_2 = [(base_row, i) for i in range(7, 4, -1)]
        layout = part_0 + part_1 + part_2
        for i in range(5, len(layout)):
            if self.__board[layout[i][0]][layout[i][1]] == self.__opponent:
                for j in range(1,5):
                    if i+j <= 12 and self.__board[layout[i+j][0]][layout[i+j][1]] == self.__turn:
                        potential += -16 * self.__probabilities[j]
        return potential

    def eat_potetntial(self):
        potential = 0
        base_row = self.get_base_row()
        part_0 = [(base_row, i) for i in range(4, -1, -1)]
        part_1 = [(1, i) for i in range(8)]
        part_2 = [(base_row, i) for i in range(7, 4, -1)]
        layout = part_0 + part_1 + part_2
        for i in range(1, len(layout)-3):
            if self.__board[layout[i][0]][layout[i][1]] == self.__turn:
                for j in range(1,5):
                    if i+j <= 12 and self.__board[layout[i+j][0]][layout[i+j][1]] == self.__opponent:
                        potential += 16 * self.__probabilities[j]
        return potential



    # def calculate_reward(self):
    #     """
    #     a function which takes the board and calculates a reward for a certain player
    #     :parameter player: which player to calculate the reward for.
    #     :return: a floating point number
    #     """
    #
    #     if self.__prev_board == None:
    #         return 0
    #     score = 0
    #     player = self.current_player()
    #     if player == 'W':
    #         return 0
    #     #     # was a white piece eaten?
    #     #     if self.__board[0][4] > self.__prev_board[BOARD][0][4]:
    #     #         score += -5
    #     #     # did a white piece exit the board?
    #     #     if self.__board[0][5] > self.__prev_board[BOARD][0][5]:
    #     #         score += 10
    #     #     # was there a double turn for the player?
    #     #     if self.current_player() == self.__prev_board[PLAYER]:
    #     #         score += 5
    #     #     # did a black piece get eaten?
    #     #     if self.__board[2][4] > self.__prev_board[BOARD][2][4]:
    #     #         score += 6
    #     #     # did a black piece exit the board?
    #     #     if self.__board[2][5] > self.__prev_board[BOARD][2][5]:
    #     #         score += -5
    #     #     # control of middle rosette - white
    #     #     if self.__board[1][3] == self.current_player():
    #     #         score += 5
    #     #     # control of middle rosette - black
    #     #     if self.__board[1][3] == self.current_opponent():
    #     #         score += -5
    #     # black player
    #     else:
    #         # was a black piece eaten?
    #         if self.__board[2][4] > self.__prev_board[BOARD][0][4]:
    #             score += -10
    #         # did a black piece exit the board?
    #         if self.__board[2][5] > self.__prev_board[BOARD][0][5]:
    #             score += 10
    #         # was there a double turn for the player?
    #         # if self.current_player() == self.__prev_board[PLAYER]:
    #         #     score += 10
    #         # did a white piece get eaten?
    #         if self.__board[0][4] > self.__prev_board[BOARD][2][4]:
    #             score += 10
    #         # did a white piece exit the board?
    #         if self.__board[0][5] > self.__prev_board[BOARD][2][5]:
    #             score += -10
    #         # control of middle rosette - black
    #         # if self.__board[1][3] == self.current_player():
    #         #     score += 10
    #         # control of middle rosette - white
    #         if self.__board[1][3] == self.current_opponent():
    #             score += -10
    #         # did a black piece exit the board?
    #         # if self.__board[2][5] > self.__prev_board[BOARD][2][5]:
    #         #     score += 10
    #
    #         # number of my pieces on board
    #         # score += (7 - self.__board[2][5] - self.__board[2][4]) * 1
    #         # score += (7 - self.__board[0][5] - self.__board[0][4]) * -2
    #
    #         # potential for double turn
    #         # score += self.double_turn_potetntial()
    #
    #         # potential for being eaten
    #         # score += self.be_eaten_potential()
    #
    #         # potential to eat
    #         # score += self.eat_potetntial()
    #
    #
    #     return score

    def calculate_reward(self):
        """
        a function which takes the board and calculates a reward for a certain player
        :parameter player: which player to calculate the reward for.
        :return: a floating point number
        """
        if self.__prev_board == None:
            return 0
        score = 0
        player = self.current_player()
        if player == 'W':
            # was a white piece eaten?
            if self.__board[0][4] > self.__prev_board[BOARD][0][4]:
                score += -5
            # did a white piece exit the board?
            if self.__board[0][5] > self.__prev_board[BOARD][0][5]:
                score += 10
            # was there a double turn for the player?
            if self.current_player() == self.__prev_board[PLAYER]:
                score += 5
            # did a black piece get eaten?
            if self.__board[2][4] > self.__prev_board[BOARD][2][4]:
                score += 6
            # did a black piece exit the board?
            if self.__board[2][5] > self.__prev_board[BOARD][2][5]:
                score += -5
            # control of middle rosette - white
            if self.__board[1][3] == self.current_player():
                score += 5
            # control of middle rosette - black
            if self.__board[1][3] == self.current_opponent():
                score += -5
        # black player
        else:
            # was a black piece eaten?
            if self.__board[2][4] > self.__prev_board[BOARD][2][4]:
                score += -5
            # # did a black piece exit the board?
            if self.__board[2][5] > self.__prev_board[BOARD][2][5]:
                score += 10
            # was there a double turn for the player?
            # if self.current_player() == self.__prev_board[PLAYER]:
            #     score += 5
            # did a white piece get eaten?
            if self.__board[0][4] > self.__prev_board[BOARD][0][4]:
                score += 25
            # did a white piece exit the board?
            if self.__board[0][5] > self.__prev_board[BOARD][0][5]:
                score += -10
            # control of middle rosette - black
            if self.__board[1][3] == self.current_player():
                score += 5
            # control of middle rosette - white
            if self.__board[1][3] == self.current_opponent():
                score += -5
            pieces_dif = abs(self.__board[2][5] - self.__board[0][5])
            if self.__board[0][5] == 7:
                score += -10 * pieces_dif
            if self.__board[2][5] == 7:
                score += 10 * pieces_dif
            # discourage being in warzone
            for i in range(8):
                # encourage safety
                if self.__board[2][i] == self.current_player():
                    score += 1
                if i == 3:
                    continue
                if self.__board[1][i] == self.current_player():
                    score += -5 if i > 3 else -1
                if self.__board[2][i] == self.current_player():
                    score += 1
        return score


    def __run_game_q_learning(self, gui_on,selection_index_b, selection_index_w):
        """

        :return:
        """

        white_roll = 0
        black_roll = 0
        while white_roll == black_roll:
            white_roll = sum(self.roll_dice())
            black_roll = sum(self.roll_dice())
        self.__turn, self.__opponent = ('W', 'B') if white_roll > black_roll else ('B', 'W')
        # self.__opponent = 'W' if self.__turn == 'B' else 'W'
        current_player = self.__player_W if self.__turn == 'W' else self.__player_B
        if gui_on:
            GUI = gui.GUI(self)
            GUI.set_black_player('', selection_index_b)
            GUI.set_white_player('', selection_index_w)
        winner = None
        while not winner:
            winner = self.is_game_over()
            if winner:
                if isinstance(self.__player_W, Agents.DeepQAgent) and self.__player_W.learning_on:
                    self.__player_W.ending_addition(self, reward=self.calculate_reward())
                    self.__player_W.learn()
                if isinstance(self.__player_B, Agents.DeepQAgent) and self.__player_B.learning_on:
                    self.__player_B.ending_addition(self, reward=self.calculate_reward())
                    self.__player_B.learn()
                break
            chosen_action = current_player.get_action(self, reward=self.calculate_reward())
            if not chosen_action:
                current_player = self.__change_turn()
                continue
            board_before_move = self.get_current_board()
            if gui_on:
                time.sleep(0.05)  # simulate thinking
                GUI.draw_board(self, self.__turn)

            if isinstance(current_player, Agents.DeepQAgent):
                self.__prev_board = (copy.deepcopy(self.__board), self.current_player())
            new_position = self.__move_piece(*chosen_action)

            while self.__is_rosette(*new_position):
                chosen_action = current_player.get_action(self, reward=self.calculate_reward())
                if not chosen_action:
                    break
                if isinstance(current_player, Agents.DeepQAgent):
                    self.__prev_board = (copy.deepcopy(self.__board), self.current_player())
                new_position = self.__move_piece(*chosen_action)
                if gui_on:
                    time.sleep(0.05)  # simulate thinking
                    GUI.draw_board(self, self.__turn)

            current_player = self.__change_turn()
        return winner

def determine_player(arg, expectimax_depth=1, black_player=False):
    """
    Instantiates the appropriate agent/ player for the game.
    :param arg: Index of player to isntantiate.
    :type arg: int
    :param expectimax_depth: Depth of expectimax algorithm, if needed
    :type expectimax_depth: int
    :param black_player: Determine the color of the player
    :type black_player: boolean
    :return: Instantiated player, and respective index
    :rtype (Agents.py class, int)
    """

    index, opponent_index = ('B','W') if black_player else ('W','B')
    if not expectimax_depth:
        expectimax_depth = 1
    if arg == 1:
        player = Agents.Human()
    elif arg == 2:
        player = expectiminimax_agent.ExpectiminimaxAgent(index, opponent_index, expectimax_depth)
    elif arg == 3:
        player = Agents.DeepQAgent(index,learning_on=args.learning_white if index == 'W' else args.learning_black,
        load_path=args.load_path_white if index == 'W' else args.load_path_black, save_path=args.save_path_white
            if index == 'W' else args.save_path_black)
    elif arg == 4:
        player = simpleAgents.RandomAgent(index)
    elif arg == 5:
        player = simpleAgents.GreedyAgent(index)
    elif arg == 6:
        player = Agents.DeepQV2(index,learning_on=args.learning_white if index == 'W' else args.learning_black,
        load_path=args.load_path_white if index == 'W' else args.load_path_black, save_path=args.save_path_white
            if index == 'W' else args.save_path_black)
    else:
        player = simpleAgents.RandomAgent(index)

    return player, arg

def multi_game_runner(args):
    cur_time = time.time()
    count_w = 0
    count_b = 0
    num_of_games = args.num_of_games
    total_black_wins = 0
    total_white_wins = 0
    last_ten_w = 0
    last_ten_b = 0
    games_played = 0
    black_player, selection_index_b = determine_player(args.black_player,
                                                       expectimax_depth=args.expectimax_depth_b,
                                                       black_player=True)  # TODO Q AGENT NOT INSTANTIATED
    white_player, selection_index_w = determine_player(args.white_player,
                                                       expectimax_depth=args.expectimax_depth_w, black_player=False)
    for i in range(num_of_games):
        b = Board()
        b._Board__player_W = white_player
        b._Board__player_B = black_player
        winner = b._Board__run_game_q_learning(args.gui_on, selection_index_b, selection_index_w)
        if winner == 'W':
            print("Winner is White")
            count_w += 1
            last_ten_w += 1
            total_white_wins += 1
        else:
            count_b += 1
            last_ten_b += 1
            total_black_wins += 1
            print("Winner is Black")
        games_played += 1
        if games_played % 10 == 0:
            print("total games played: {}, in the last 10 games black won {} times and white won {} times".format(
                games_played, last_ten_b, last_ten_w))
            last_ten_w, last_ten_b = (0, 0)
            if isinstance(b._Board__player_B, Agents.DeepQV2) and b._Board__player_B.learning_on:
                b._Board__player_B.network.save(b._Board__player_B.save_path)
            if isinstance(b._Board__player_W, Agents.DeepQV2) and b._Board__player_W.learning_on:
                b._Board__player_W.network.save(b._Board__player_W.save_path)
            if type(b._Board__player_B) is Agents.DeepQAgent and b._Board__player_B.learning_on:
                b._Board__player_B.network.save_weights(b._Board__player_B.save_path)
            if type(b._Board__player_W) is Agents.DeepQAgent and b._Board__player_W.learning_on:
                b._Board__player_W.network.save_weights(b._Board__player_W.save_path)


    if isinstance(b._Board__player_B, Agents.DeepQV2) and b._Board__player_B.learning_on and b._Board__player_B.save_path != None:
        b._Board__player_B.network.save(b._Board__player_B.save_path)
    if isinstance(b._Board__player_W, Agents.DeepQV2) and b._Board__player_W.learning_on and b._Board__player_W.save_path != None:
        b._Board__player_W.network.save(b._Board__player_W.save_path)
    if type(b._Board__player_B) is Agents.DeepQAgent and b._Board__player_B.learning_on:
        b._Board__player_B.network.save_weights(b._Board__player_B.save_path)
    if type(b._Board__player_W) is Agents.DeepQAgent and b._Board__player_W.learning_on:
        b._Board__player_W.network.save_weights(b._Board__player_W.save_path)
    print("time elapsed is for {} games is {}".format(num_of_games, time.time() - cur_time))
    print("White won: " + str(count_w))
    print("Black won: " + str(count_b))


def single_game_runner(args):

    board = Board()
    black_player = simpleAgents.RandomAgent('B')
    white_player = simpleAgents.RandomAgent('W')
    if not args.main_menu:
        GUI = gui.GUI(board, main_menu=True)
    else:
        GUI = gui.GUI(board)
        black_player, selection_index_b = determine_player(args.black_player,
                                                           expectimax_depth=args.expectimax_depth_b,
                                                           black_player=True)  # TODO Q AGENT NOT INSTANTIATED
        white_player, selection_index_w = determine_player(args.white_player,
                                                           expectimax_depth=args.expectimax_depth_w, black_player=False)
        GUI.set_black_player('', selection_index_b)
        GUI.set_white_player('', selection_index_w)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mm','--main_menu', action='store_true',help=MAIN_MENU_HELP)
    parser.add_argument('-b', '--black_player', type=int, help=BLACK_PLAYER_HELP)
    parser.add_argument('-w', '--white_player', type= int, help=WHITE_PLAYER_HELP)
    parser.add_argument('-depth_b', '--expectimax_depth_b', type=int, help=EXPECTIMAX_DEPTH_HELP)
    parser.add_argument('-depth_w', '--expectimax_depth_w', type=int,
                        help=EXPECTIMAX_DEPTH_HELP)
    parser.add_argument('-num_of_games', type=int, help=NUM_OF_GAMES_HELP)
    parser.add_argument('-delay', '--delay', type=float, help=DELAY_HELP)
    parser.add_argument('-gm', '--game_mode' ,type=int, help=GAME_MODE_HELP, default=0)
    parser.add_argument('-lpb', '--load_path_black', type=str, help=LOAD_PATH_HELP, default='best_player_yet')
    parser.add_argument('-lpw', '--load_path_white', type=str, help=LOAD_PATH_HELP, default='best_player_yet')
    parser.add_argument('-spb', '--save_path_black', type=str, help=SAVE_PATH_HELP, default=None)
    parser.add_argument('-spw', '--save_path_white', type=str, help=SAVE_PATH_HELP, default=None)
    parser.add_argument('-lw', '--learning_white', type=int, help=LEARNING_HELP, default=False)
    parser.add_argument('-lb', '--learning_black', type=int, help=LEARNING_HELP, default=False)
    parser.add_argument('-gui', '--gui_on', type=int, help=GUI_HELP, default=False)

    args = parser.parse_args()
    if args.game_mode == 1:
        multi_game_runner(args)

    if args.game_mode == 0:
        single_game_runner(args)
