from util import raiseNotDefined
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, InputLayer, Flatten
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
from random import choices
from collections import deque
import util as util
from game_engine_experiment import Board
import copy
import os

# CONSTANTS
BOARD_HEIGHT = 3
BOARD_WIDTH = 8
NUMBER_OF_PLAYERS = 2
BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT - 4
EPSILON = 0.8
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
DISCOUNT = 0.5
LEARNING_RATE = 0.01
DB_SIZE = 20
BATCH_SIZE = 10


class DeepQAgent:

    def __init__(self, index):
        self.index = index

        self.learning_rate = LEARNING_RATE
        self.network = self.build_network()
        self.epsilon = EPSILON
        self.discount = DISCOUNT
        self.db = deque(maxlen=DB_SIZE)
        self.batch_size = BATCH_SIZE
        self.iteration_counter = 0
        self.prev_state = None
        self.prev_action = None
        self.turn_count = 0

        # initial fit to speed up learning in game start todo: check if initial fit is needed
        # rand_state = np.random.rand(1, 1, BOARD_SIZE + NUMBER_OF_PLAYERS)
        # rand_reward = np.random.rand(1, 1, 1)
        # self.network.predict(rand_state)
        # self.network.fit(rand_state, rand_reward, epochs=1, verbose=0)

    def build_network(self):
        """
        setup NN architecture for the policy and return it
        :return:
        """
        model = Sequential()

        # First 20 nodes in the input represent the board state while the last 4 represent how
        # many pieces player 1 and 2 have out of the game, or not yet on the board.
        model.add(InputLayer(input_shape=(1,24)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mae'])
        return model

    def get_action(self, state: Board, reward):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """

        # Pick Action

        self.turn_count += 1
        dice_roll = state.roll_dice()
        legal_moves = state.get_legal_moves()
        dice_sum = sum(dice_roll)
        # state_vector = self.create_nn_input_state(state)
        state_vector = self.create_nn_input(state.current_player(), np.array(copy.deepcopy(state.get_current_board())))

        if dice_sum == 0 or not legal_moves:
            return None

        # can't learn from the first move, so let it be random.
        if self.prev_state is None:
            # update prev action to a random action and return this as the current action
            # also update the state to the current state.
            self.prev_state = state_vector
            self.prev_action = random.choice(legal_moves)
            return self.prev_action

        # We need to record our actions as tuples = (prev_state, prev_action, reward, cur_state)
        # and add these actions to our database

        db_tup = (self.prev_state, self.prev_action, reward, state_vector, legal_moves, dice_sum, state.get_current_board())
        self.db.append(db_tup)

        # Decide on action being exploration or exploitation
        # exploration
        if util.flipCoin(self.epsilon):
            action = random.choice(legal_moves)
        # exploitation
        else:
            action = self.choose_action(copy.deepcopy(state.get_current_board()), legal_moves, dice_sum)

        # update previous state and action to be the ones being taken right now. then return the action
        self.prev_state = state_vector
        self.prev_action = action

        if self.turn_count % 10 == 0:
            self.learn()
        return action

    def choose_action(self, state_vector, legal_moves, dice_sum):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        from game_engine import Board
        successor_states = Board.get_successors(state_vector, self.index)
        successor_states_list = successor_states.get(dice_sum)
        successor_states_list_reworked = [self.create_nn_input(successor[1], np.array(successor[0])) for successor in successor_states_list]
        action_scores = [self.get_q_value(successor_state) for successor_state in successor_states_list_reworked]
        max_indices = np.argmax(action_scores)
        if type(max_indices) == np.int64:
            return legal_moves[max_indices]
        else:
            return legal_moves[random.choice(max_indices)]

        # best_action = None
        # best_action_score = -float('inf')
        # for action in legal_moves:
        #     cur_q_val = self.get_q_value(state, action)
        #     if cur_q_val > best_action_score:
        #         best_action = action
        #         best_action_score = cur_q_val
        #     # random tiebreaker in case values are equal.
        #     elif cur_q_val == best_action_score:
        #         best_action = random.choice([best_action, action])
        # return best_action


    def get_q_value(self, successor_state):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        # all_actions = []
        # all_actions.append(self.create_nn_input(state, action).reshape(1, -1))
        # scores = self.network.predict(np.array(all_actions))
        # state_vector = state_vector
        # action_vector = self.create_nn_input_action(action)
        # concat_vector = self.create_nn_input_concat(state_vector, action_vector)
        scores = self.network.predict(successor_state)
        return scores

    def create_nn_input_action(self, action):
        action_vector = np.zeros((3, 8))
        action_vector[action[0], action[1]] = 1
        action_vector = action_vector.reshape((1,-1))
        return action_vector

    def create_nn_input_state(self, state: Board):
        """
        creates an input vector for the neural network according to the given state and action
        :param state: board representation as in game engine Board
        :type state: list of lists
        :param action: (row, column) as in Board API
        :type action: tuple
        :return: input vector for the network
        :rtype:
        """
        my_color = state.current_player()
        opponent_color = state.current_opponent()
        my_base = state.get_base_row()
        opponent_base = state.get_base_row_opponent()
        board = np.array(state.get_current_board())
        dice_roll = sum(state.get_current_dice())

        my_pieces_left = np.array([board[my_base][4]]).astype(np.int64).reshape((1,-1))
        my_pieces_out = np.array([board[my_base][5]]).astype(np.int64).reshape((1,-1))
        board_by_me = np.where(board == my_color, 1, 0).reshape((1,-1))

        opponent_pieces_left = np.array([board[opponent_base][4]]).astype(np.int64).reshape((1,-1))
        opponent_pieces_out = np.array([[board[opponent_base][5]]]).astype(np.int64).reshape((1,-1))
        board_by_opponent = np.where(board == opponent_color, 1, 0).reshape((1,-1))

        dice_vector = np.zeros((1,5)).reshape((1,-1))
        dice_vector[:,dice_roll] = 1

        return np.hstack([my_pieces_left,
                          my_pieces_out,
                          board_by_me,
                          opponent_pieces_left,
                          opponent_pieces_out,
                          board_by_opponent,
                          dice_vector])

    def create_nn_input_concat(self, state_vec, action_vec):
        return np.hstack([state_vec,action_vec])

    def create_nn_input(self, current_player, state_copy):
        """
        fun tries convert np array of strings to the relevant ints
        :param state:
        :return:
        """
        current_opponent = 'B' if current_player == 'W' else 'W'
        state_copy[np.where(state_copy == '_')] = 0
        state_copy[np.where(state_copy == current_opponent)] = 2
        state_copy[np.where(state_copy == current_player)] = 1
        state_vector = state_copy.astype(np.int)
        state_vector[np.where(state_vector == 2)] = -1
        return state_vector.reshape(1,-1)

    def learn(self):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
        """
        # self.cur_learner(round,prev_state,prev_action, reward,new_state, too_slow)
        if len(self.db) < self.batch_size:
            return

        nn_input_array = []
        reward_array = []

        samples = choices(self.db, k=self.batch_size)
        for state_tuple in samples:
            prev_state_vector, prev_action, reward, next_state_vector, legal_moves, dice_sum, next_vector_untouched = state_tuple
            best_action_score = self.get_max_value(next_vector_untouched, legal_moves, dice_sum)
            target = reward + self.discount * best_action_score

            nn_input_array.append(np.array(prev_state_vector).reshape(1, -1))
            reward_array.append(target)

        reward_array = np.array(reward_array)[:, np.newaxis, np.newaxis]
        nn_input_array = np.array(nn_input_array)
        self.network.fit(nn_input_array, reward_array, epochs=1, verbose=0)
        # update epsilon to make it lower

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def get_max_value(self, state_vector, legal_moves, dice_sum):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value None
        """
        from game_engine import Board
        if not legal_moves:  # terminal state case
            return 0
        else:
            successor_states = Board.get_successors(state_vector, self.index)
            successor_states_list = successor_states.get(dice_sum)
            successor_states_list_reworked = [self.create_nn_input(successor[1], np.array(successor[0])) for successor in
                                              successor_states_list]
            action_scores = [self.get_q_value(successor_state) for successor_state in successor_states_list_reworked]
            max_score = np.max(action_scores)
            return max_score
