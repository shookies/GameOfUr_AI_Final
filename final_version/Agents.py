import random
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from random import choices
from collections import deque
import util as util
from game_engine import Board
import pickle
import time

# CONSTANTS
BOARD_HEIGHT = 3
BOARD_WIDTH = 8
NUMBER_OF_PLAYERS = 2
BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT - 4
EPSILON = 0.99
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
DISCOUNT = 0.9
LEARNING_RATE = 0.001
DB_SIZE =64
BATCH_SIZE = 32


class DeepQAgent:

    def __init__(self, index, learning_on=False, load_path='best_player_yet', save_path=None):
        self.index = index

        self.learning_on = learning_on
        self.learning_rate = LEARNING_RATE
        self.network = self.build_network()
        self.network.load_weights(load_path)
        self.save_path = save_path

        self.epsilon = EPSILON
        self.discount = DISCOUNT
        self.db = deque(maxlen=DB_SIZE)
        self.batch_size = BATCH_SIZE
        self.iteration_counter = 0
        self.prev_state = None
        self.prev_action = None
        self.turn_count = 0


    def build_network(self):
        """
        setup NN architecture for the policy and return it
        :return:
        """
        model = Sequential()
        model.add(InputLayer(input_shape=(1,61)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=["mse"])
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
        state_vector = self.create_nn_input_state(state)
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

        db_tup = (self.prev_state, self.prev_action, reward, state_vector, legal_moves)
        self.db.append(db_tup)

        # Decide on action being exploration or exploitation
        # exploration
        if self.learning_on and util.flipCoin(self.epsilon):
            action = random.choice(legal_moves)
        # exploitation
        else:
            action = self.choose_action(state_vector, legal_moves)

        # update previous state and action to be the ones being taken right now. then return the action
        self.prev_state = state_vector
        self.prev_action = action

        if self.turn_count % 32 == 0 and self.learning_on:
            self.learn()
        return action

    def choose_action(self, state_vector, legal_moves):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        # TODO delete up to here - trying to make more efficient
        vectors = np.vstack([self.create_nn_input_concat(state_vector, self.create_nn_input_action(move))for move in legal_moves])
        max_indices = np.argmax(self.get_q_value_vectorized(vectors))
        if type(max_indices) == np.int64:
            return legal_moves[max_indices]
        else:
            return legal_moves[random.choice(max_indices)]


    def get_q_value_vectorized(self, vectors):
        scores = self.network.predict(vectors, batch_size=len(vectors))
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
        my_base, enemy_base = (0,2) if self.index == 'W' else (2,0)
        enemy_index = 'B' if self.index == 'W' else 'W'
        board = state.get_current_board()
        my_base_row = []
        for i in board[my_base]:
            if i == '_':
                my_base_row.append(0)
                continue
            if i == self.index:
                my_base_row.append(1)
                continue
            my_base_row.append(i)
        enemy_base_row = []
        for i in board[enemy_base]:
            if i == '_':
                my_base_row.append(0)
                continue
            if i == enemy_index:
                my_base_row.append(1)
                continue
            my_base_row.append(i)
        mid_alley = []
        for i in board[1]:
            if i == self.index:
                mid_alley.append([1,0])
                continue
            if i == enemy_index:
                mid_alley.append([0,1])
                continue
            mid_alley.append([0,0])


        dice_roll = state.get_current_dice()
        dice_vector = np.zeros((1,5)).reshape((1,-1))
        dice_vector[:,dice_roll] = 1
        return np.hstack([np.array(my_base_row).reshape((1,-1)),
                          np.array(enemy_base_row).reshape((1,-1)),
                          np.array(mid_alley).reshape((1,-1)),
                          dice_vector])

    def create_nn_input_concat(self, state_vec, action_vec):
        return np.hstack([state_vec,action_vec])


    def learn(self):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
        """

        if len(self.db) < self.batch_size:
            return

        nn_input_array = []
        reward_array = []

        samples = choices(self.db, k=self.batch_size)
        for state_tuple in samples:
            prev_state_vector, prev_action, reward, next_state_vector ,legal_moves= state_tuple
            best_action_score = self.get_max_value(next_state_vector, legal_moves)
            target = reward + self.discount * best_action_score

            nn_input_array.append(self.create_nn_input_concat(prev_state_vector, self.create_nn_input_action(prev_action)).reshape(1, -1))
            reward_array.append(target)

        reward_array = np.array(reward_array)[:, np.newaxis, np.newaxis]
        nn_input_array = np.array(nn_input_array)
        self.network.fit(nn_input_array, reward_array, epochs=10, verbose=0)
        # update epsilon to make it lower

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
        # print(self.epsilon)

    def get_max_value(self, state_vector, legal_moves):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value None
        """
        if not legal_moves:  # terminal state case
            return 0
        else:
            vectors = np.vstack(
                [self.create_nn_input_concat(state_vector, self.create_nn_input_action(move)) for move in legal_moves])
            max_score = np.max(self.get_q_value_vectorized(vectors))
            return max_score


    def ending_addition(self, state: Board, reward):
        """
        add last move to the DB when the game ends.
        """
        state_vector = self.create_nn_input_state(state)
        db_tup = (self.prev_state, self.prev_action, reward, state_vector, [])
        self.db.append(db_tup)
        return




class DeepQV2(DeepQAgent):

    def __init__(self, index, learning_on=False, load_path='best_player_yet', save_path=None):
        self.index = index

        self.learning_on = learning_on
        self.learning_rate = 0.005
        self.network = self.build_network()
        self.network = load_model(load_path)
        self.save_path = save_path

        self.epsilon = EPSILON
        self.discount = DISCOUNT
        self.db = deque(maxlen=100)
        self.batch_size = 16
        self.iteration_counter = 0
        self.prev_state = None
        self.prev_action = None
        self.turn_count = 0

    def build_network(self):
        """
        setup NN architecture for the policy and return it
        :return:
        """
        model = Sequential()
        model.add(InputLayer(input_shape=(1, 81)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mae'])
        return model

    def create_nn_input_action(self, action):
        action_vector = np.zeros((3, 8))
        action_vector[action[0], action[1]] = 1
        action_vector = action_vector.reshape((1, -1))
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

        my_pieces_left = np.array([board[my_base][4]]).astype(np.int64).reshape((1, -1))
        my_pieces_out = np.array([board[my_base][5]]).astype(np.int64).reshape((1, -1))
        board_by_me = np.where(board == my_color, 1, 0).reshape((1, -1))

        opponent_pieces_left = np.array([board[opponent_base][4]]).astype(np.int64).reshape((1, -1))
        opponent_pieces_out = np.array([[board[opponent_base][5]]]).astype(np.int64).reshape((1, -1))
        board_by_opponent = np.where(board == opponent_color, 1, 0).reshape((1, -1))

        dice_vector = np.zeros((1, 5)).reshape((1, -1))
        dice_vector[:, dice_roll] = 1

        return np.hstack([my_pieces_left,
                          my_pieces_out,
                          board_by_me,
                          opponent_pieces_left,
                          opponent_pieces_out,
                          board_by_opponent,
                          dice_vector])

    def create_nn_input_concat(self, state_vec, action_vec):
        return np.hstack([state_vec, action_vec])


class DeepQOffline:

    def __init__(self, index):
        self.index = index
        self.learning_rate = LEARNING_RATE
        self.network = self.build_network()
        self.epsilon = EPSILON
        self.discount = DISCOUNT
        self.batch_size = 128
        self.iteration_counter = 0
        self.learn_iterations = 1000
        self.database = None
        # self.load_database()

    def load_database(self):
        start = time.time()
        print("loading database")
        def load_all():
            with open('expectimax_games_db.pkl', 'rb') as db:
                # while True:
                for i in range(100000):
                # while True:
                    try:
                        yield pickle.load(db)
                    except EOFError:
                        break

        self.database = list(load_all())
        print("done loading, " + str(time.time() - start))



    def calc_sum_distances(self, board, base_row, player):
        distances = []
        part_0 = [(base_row, i) for i in range(4, -1, -1)]
        part_1 = [(1, i) for i in range(8)]
        part_2 = [(base_row, i) for i in range(7, 4, -1)]
        layout = part_0 + part_1 + part_2
        for i in range(len(layout)-1):
            if board[layout[i][0]][layout[i][1]] == player:
                distances.append(15-i)
        distances.append(15 * board[base_row][4])
        return sum(distances)

    def calculate_reward(self, prev_board:Board, next_board:Board, who_played):
        """
        a function which takes the board and calculates a reward for a certain player
        :parameter player: which player to calculate the reward for.
        :return: a floating point number
        """
        # reward_multiplyier = 1 if who_played == self.index else -1
        my_base, enemy_base = (0, 2) if who_played == 'W' else (2, 0)

        score = 0

        prev_board_list = prev_board.get_current_board()
        next_board_list = next_board.get_current_board()
        # # enemy was eaten
        # if prev_board_list[enemy_base][4] < next_board_list[enemy_base][4]:
        #     score += 10
        # # my piece was eaten
        # if prev_board_list[my_base][4] < next_board_list[my_base][4]:
        #     score -= 10
        # # my piece left the board
        # if prev_board_list[my_base][5] < next_board_list[my_base][5]:
        #     score += 10
        # # enemy piece left the board
        # if prev_board_list[my_base][5] < next_board_list[my_base][5]:
        #     score -= 10
        # # double turn for me todo how to find this out without passing more variables
        #
        # # i kept or gained control of the middle rosette:
        # # if next_board_list[1][3] == who_played:
        # #     score += 5
        # if next_board_list[1][3] != who_played:
        #     score -= 20
        # # i won
        # # if next_board_list[my_base][5] == 7:
        # #     score += 50
        # # if next_board_list[enemy_base][5] == 7:
        # #     score -= 50
        # # score = (next_board_list[my_base][5] - next_board_list[enemy_base][5]) ** 3
        score += (self.calc_sum_distances(prev_board_list, 2, 'B') - self.calc_sum_distances(next_board_list, 2,
                                                                                                    'B')) / 3
        score -= (self.calc_sum_distances(prev_board_list, 0, 'W') - self.calc_sum_distances(next_board_list, 0,
                                                                                                    'W')) / 3
        return score


    def build_network(self):
        """
        setup NN architecture for the policy and return it
        :return:
        """
        print("building network")
        start = time.time()
        model = Sequential()

        # First 20 nodes in the input represent the board state while the last 4 represent how
        # many pieces player 1 and 2 have out of the game, or not yet on the board.
        model.add(InputLayer(input_shape=(1,53)))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mae'])
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mse'])
        print("done building netwrok, " + str(time.time() - start))
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
        dice_roll = state.roll_dice()
        legal_moves = state.get_legal_moves()
        dice_sum = sum(dice_roll)
        if dice_sum == 0 or not legal_moves:
            return None

        state_vector = self.create_nn_input_state(state, self.index)
        # Decide on action being exploration or exploitation
        # exploration
        if util.flipCoin(self.epsilon):
            action = random.choice(legal_moves)
        # exploitation
        else:
            action = self.choose_action(state_vector, legal_moves)

        return action

    def choose_action(self, state_vector, legal_moves):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        vectors = np.vstack([self.create_nn_input_concat(state_vector, self.create_nn_input_action(move))
                             for move in legal_moves])
        max_indices = np.argmax(self.get_q_value_vectorized(vectors))
        if type(max_indices) == np.int64:
            return legal_moves[max_indices]
        else:
            return legal_moves[random.choice(max_indices)]

    def get_q_value_vectorized(self, vectors):
        scores = self.network.predict(vectors, batch_size=len(vectors))
        return scores

    def create_nn_input_action(self, action):
        # my_base_row = 0 if self.index == 'W' else 2
        # todo this way the agents sees it's base as the first half as the vector, and the mid alley as the second
        # todo agents doesn't see the other player's row since it is irrelevant
        row = 0 if action[0] != 1 else 1
        action_vector = np.zeros((2, 8))
        action_vector[row, action[1]] = 1
        action_vector = action_vector.reshape((1,-1))
        return action_vector

    def load_network(self):
        self.network.load_weights("deep_q_offline_weights")

    def preproccess_data(self):
        print("preporocessing data")
        start = time.time()
        state_vectors = []
        action_vectors = []
        labels = []
        legal_moves = []

        for item in self.database:

            prev_board, next_board, player, action = item
            # if player == self.index:
            reward = self.calculate_reward(prev_board, next_board, player)
            state_vector = self.create_nn_input_state(prev_board, player)
            action_vector = self.create_nn_input_action(action)

            # vectors.append(self.create_nn_input_concat(state_vector, action_vector))
            state_vectors.append(state_vector)
            action_vectors.append(action_vector)
            labels.append(reward)
            legal_moves.append(prev_board.get_legal_moves())
            # legal_moves.append(next_board.get_legal_moves())

        print("done preprocessing, " + str(time.time() - start))
        return np.array(state_vectors), np.array(labels), np.array(legal_moves), np.array(action_vectors)


    def create_nn_input_state(self, state: Board, player):
        """
        creates an input vector for the neural network according to the given state and action
        :param state: board representation as in game engine Board
        :type state: list of lists
        :param action: (row, column) as in Board API
        :type action: tuple
        :return: input vector for the network
        :rtype:
        """
        my_base, enemy_base = (0,2) if player == 'W' else (2,0)
        enemy_index = 'B' if player == 'W' else 'W'
        board = state.get_current_board()
        my_base_row = []
        for i in board[my_base]:
            if i == '_':
                my_base_row.append(0)
                continue
            if i == player:
                my_base_row.append(1)
                continue
            my_base_row.append(i)
        enemy_base_row = []
        for i in board[enemy_base]:
            if i == '_':
                my_base_row.append(0)
                continue
            if i == enemy_index:
                my_base_row.append(1)
                continue
            my_base_row.append(i)
        mid_alley = []
        for i in board[1]:
            if i == player:
                mid_alley.append([1,0])
                continue
            if i == enemy_index:
                mid_alley.append([0,1])
                continue
            mid_alley.append([0,0])


        dice_roll = state.get_current_dice()
        dice_vector = np.zeros((1,5)).reshape((1,-1))
        dice_vector[:,dice_roll] = 1
        return np.hstack([np.array(my_base_row).reshape((1,-1)),
                          np.array(enemy_base_row).reshape((1,-1)),
                          np.array(mid_alley).reshape((1,-1)),
                          dice_vector])



    def create_nn_input_concat(self, state_vec, action_vec):
        return np.concatenate([state_vec,action_vec], axis=-1)


    def learn(self):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
        """
        # self.cur_learner(round,prev_state,prev_action, reward,new_state, too_slow)
        state_vectors, labels, legal_actions, action_vectors = self.preproccess_data()
        state_vectors, labels, action_vectors = state_vectors[:-1], labels[:-1], action_vectors[:-1]
        next_state_vectors, next_legal_action = state_vectors[1:], legal_actions[1:]
        legal_actions = np.array(legal_actions, dtype=np.object)
        print("starting learning iterations")
        start = time.time()
        for i in range(1000):
            if i%100 == 0:
                print("iteration " + str(i) + ", time per " + str(i) + " iterations: " + str(time.time() - start))

            samples = choices(range(len(next_state_vectors)), k=self.batch_size)
            best_action_scores = self.get_max_value(next_state_vectors[samples], legal_actions[samples].tolist())
            target = labels[samples] + self.discount * best_action_scores
            train_set = self.create_nn_input_concat(state_vectors[samples],action_vectors[samples])

            reward_array = target[:, np.newaxis, np.newaxis]
            self.network.fit(train_set, reward_array, epochs=10, verbose=0)
        self.network.save_weights("deep_q_offline_weights")
        print("done learning, "+ str(time.time()-start))



        # update epsilon to make it lower

        # if self.epsilon > EPSILON_MIN:
        #     self.epsilon *= EPSILON_DECAY


    def get_max_value(self, state_vectors, legal_moves):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value None
        """
        if not legal_moves:  # terminal state case
            return 0
        all_inputs = []
        for i in range(len(legal_moves)):
            for move in legal_moves[i]:
                all_inputs.append(self.create_nn_input_concat(state_vectors[i], self.create_nn_input_action(move)))
        vectors = np.vstack(all_inputs)
        max_score = np.max(self.get_q_value_vectorized(vectors))
        return max_score

class Human:
    """
    Dummy class in order to request action from GUI, instead from an agent
    """
    def no_op(self):
        return None