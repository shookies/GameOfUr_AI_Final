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
import Agents_expertiment as Agents
EMPTY_SQUARE = '_'
BOARD = 0
PLAYER = 1

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
        return board

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
            # did a black piece exit the board?
            if self.__board[2][5] > self.__prev_board[BOARD][2][5]:
                score += 10
            # was there a double turn for the player?
            if self.current_player() == self.__prev_board[PLAYER]:
                score += 2
            # did a white piece get eaten?
            if self.__board[0][4] > self.__prev_board[BOARD][0][4]:
                score += 5
            # did a white piece exit the board?
            if self.__board[0][5] > self.__prev_board[BOARD][0][5]:
                score += -10
            # control of middle rosette - black
            # if self.__board[1][3] == self.current_player():
            #     score += 5
            # control of middle rosette - white
            if self.__board[1][3] == self.current_opponent():
                score += -2
        return score




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

    # def do_move(self, row, column):
    #     """
    #     moves piece from given square coordinates to the next position according to __current_dice_roll
    #     :param row: current row index on board
    #     :type row: int
    #     :param col: current column index on board
    #     :type col: int
    #     :return: next position coordinates move was successful, None o.w
    #     :rtype: (int,int) or None
    #     """
    #     return self.__move_piece(row, column)

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
        self.__turn, self.__opponent = self.__opponent, self.__turn
        return self.__player_W if self.__turn == 'W' else self.__player_B

    def __run_game(self):
        white_roll = 0
        black_roll = 0
        while white_roll == black_roll:
            white_roll = sum(self.roll_dice())
            black_roll = sum(self.roll_dice())
        self.__turn, self.__opponent = ('W', 'B') if white_roll > black_roll else ('B', 'W')
        # self.__opponent = 'W' if self.__turn == 'B' else 'W'
        current_player = self.__player_W if self.__turn == 'W' else self.__player_B

        reward_on_last_move = 0

        winner = None
        while not winner:
            winner = self.is_game_over()
            if winner:
                break
            # print("loop 1")

            chosen_action = current_player.get_action(self, reward=None)
            if not chosen_action:
                current_player = self.__change_turn()
                # print(self.__turn)
                continue

            # print(self)
            new_position = self.__move_piece(*chosen_action)

            while self.__is_rosette(*new_position):
                # print("loop 2")
                chosen_action = current_player.get_action(self, reward=None)
                if not chosen_action:
                    # print(self.__turn)
                    break
                new_position = self.__move_piece(*chosen_action)
                # print(self)
            current_player = self.__change_turn()

        # print("game over, winner: " + winner)
        return winner

    def __run_game_q_learning(self):
        white_roll = 0
        black_roll = 0
        while white_roll == black_roll:
            white_roll = sum(self.roll_dice())
            black_roll = sum(self.roll_dice())
        self.__turn, self.__opponent = ('W', 'B') if white_roll > black_roll else ('B', 'W')
        # self.__opponent = 'W' if self.__turn == 'B' else 'W'
        current_player = self.__player_W if self.__turn == 'W' else self.__player_B

        winner = None
        while not winner:
            winner = self.is_game_over()

            if winner:
                break
            # print("loop 1")
            chosen_action = current_player.get_action(self, reward=self.calculate_reward())
            if not chosen_action:
                current_player = self.__change_turn()
                # print(self.__turn)
                continue
            board_before_move = self.get_current_board()

            # print(self)

            self.__prev_board = (copy.deepcopy(self.__board), self.current_player())
            new_position = self.__move_piece(*chosen_action)


            while self.__is_rosette(*new_position):
                # print("loop 2")
                chosen_action = current_player.get_action(self, reward=self.calculate_reward())
                if not chosen_action:
                    # print(self.__turn)
                    break
                self.__prev_board = (copy.deepcopy(self.__board), self.current_player())
                new_position = self.__move_piece(*chosen_action)
                # print(self)

            current_player = self.__change_turn()

        # print("game over, winner: " + winner)
        return winner



if __name__ == "__main__":
    cur_time = time.time()
    count_w = 0
    count_b = 0
    num_of_games = 5000
    learning_agent = Agents.DeepQAgent('B')
    last_ten_w = 0
    last_ten_b = 0
    games_played = 0
    for i in range(num_of_games):
        b = Board()
        b._Board__player_W = simpleAgents.RandomAgent('W')
        b._Board__player_B = learning_agent
        # b._Board__player_B = simpleAgents.RandomAgent('B')
        winner = b._Board__run_game_q_learning()
        # winner = b._Board__run_game()
        if winner == 'W':
            # print("Winner is White")
            count_w += 1
            last_ten_w += 1
        else:
            count_b += 1
            last_ten_b += 1
            # print("Winner is Black")
        games_played += 1
        if games_played % 10 == 0:
            print("total games played: {}, in the last 10 games black won {} times and white won {} times".format(games_played, last_ten_b, last_ten_w))
            last_ten_w, last_ten_b = (0, 0)

    b._Board__player_B.network.save('weights')
    print("time elapsed is for {} games is {}".format(num_of_games, time.time() - cur_time))
    print("White won: " + str(count_w))
    print("Black won: " + str(count_b))
