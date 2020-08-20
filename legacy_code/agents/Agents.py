from agents.util import raiseNotDefined
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, InputLayer, Flatten
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
from random import choices
from collections import deque
import agents.util as util


# CONSTANTS
BOARD_HEIGHT = 3
BOARD_WIDTH = 8
NUMBER_OF_PLAYERS = 2
BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT - 4
EPSILON = 0.8
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01


class Agent:
    """
    An agent must define a getAction method
    """
    def __init__(self, index=0):
        self.index = index

    def get_action(self, state, reward):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        raiseNotDefined()


class ValueEstimationAgent(Agent):
    """
    Abstract agent which assigns values to (state,action)
    Q-Values for an environment. As well as a value to a
    state and a policy given respectively by,
  """
    ####################################
    #    Override These Functions      #
    ####################################
    def getQValue(self, state, action):
        """
        Should return Q(state,action)
        """
        raiseNotDefined()

    def getValue(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

      V(s) = max_{a in actions} Q(s,a)
      """
        raiseNotDefined()

    def getPolicy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with getAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        raiseNotDefined()

    def get_action(self, state, reward):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        raiseNotDefined()


class ReinforcementAgent(ValueEstimationAgent):
    """
    Abstract Reinforcemnt Agent: A ValueEstimationAgent
    which estimates Q-Values (as well as policies) from experience
    rather than a model

    What you need to know:
        - The environment will call
          observeTransition(state,action,nextState,deltaReward),
          which will call update(state, action, nextState, deltaReward)
          which you should override.
    - Use self.getLegalActions(state) to know which actions
          are available in a state
    """

    ####################################
    #    Override These Functions      #
    ####################################

    def update(self, state, action, nextState, reward):
        """
        This class will call this function, which you write, after
        observing a transition and reward
        """
        raiseNotDefined()

    def getLegalActions(self, state):
        """
        Get the actions available for a given
        state. This is what you should use to
        obtain legal actions for a state
        """
        return self.actionFn(state)

    def observeTransition(self, state, action, nextState, deltaReward):
        """
        Called by environment to inform agent that a transition has
        been observed. This will result in a call to self.update
        on the same arguments

        NOTE: Do *not* override or call this function
    """
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)

    def startEpisode(self):
        """
      Called by environment when new episode is starting
    """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
      Called by environment when episode is done
    """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def __init__(self, actionFn=None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
        """
    actionFn: Function which takes a state and returns the list of legal actions

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
        if actionFn == None:
            actionFn = lambda state: state.get_legal_actions()
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)


class DeepQLearningAgent(ReinforcementAgent):
    """
      DeepQ-Learning Agent

      Functions you should fill in:
        - getQValue
        - getAction
        - getValue
        - getPolicy
        - learn

      Instance variables we have access to
        - self.epsilon (exploration prob)
        - self.learning_rate (learning rate)
        - self.discount (discount rate)

      Functions we should use
        - self.getLegalActions(state)
          which returns legal actions
          for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.network = self.build_network()

        self.epsilon = EPSILON
        self.discount = DISCOUNT
        self.learning_rate = LEARNING_RATE
        self.db = deque(maxlen=ITERATIONS_NUM)
        self.batch_size = BATCH_SIZE
        self.iteration_counter = 0
        self.prev_state = None
        self.prev_action = None

        # initial fit to speed up learning in game start
        rand_state = np.random.rand(1, 1, BOARD_SIZE + NUMBER_OF_PLAYERS)
        rand_reward = np.random.rand(1, 1, 1)
        self.network.predict(rand_state)
        self.network.fit(rand_state, rand_reward, epochs=1, verbose=0)

    def build_network(self):
      """
      setup NN architecture for the policy and return it
      :return:
      """
      model = Sequential()

      # First 20 nodes in the input represent the board state while the last 4 represent how
      # many pieces player 1 and 2 have out of the game, or not yet on the board.
      model.add(InputLayer(input_shape=(1, BOARD_SIZE + NUMBER_OF_PLAYERS*2)))
      model.add(Dense(26, activation='relu'))
      model.add(Dense(50, activation='relu'))
      model.add(Dense(50, activation='relu'))
      model.add(Dense(16, activation='linear'))
      model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mae'])
      return model

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        "*** YOUR CODE HERE ***"
        all_actions = []
        #TODO what should the input look like for the network?
        all_actions.append(self.create_nn_input(state, action).reshape(1, -1))
        scores = self.network.predict(np.array(all_actions))
        return scores

    def getValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)
        if not legal_actions:  # terminal state case
            return 0
        else:
            best_action_score = -float('inf')
            for action in legal_actions:
                cur_q_val = self.getQValue(state, action)
                if cur_q_val > best_action_score:
                    best_action_score = cur_q_val
            return best_action_score

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)
        if not legal_actions:  # terminal state case
            return None
        else:
            best_action = None
            best_action_score = -float('inf')
            for action in legal_actions:
                cur_q_val = self.getQValue(state, action)
                if cur_q_val > best_action_score:
                    best_action = action
                    best_action_score = cur_q_val
                # random tiebreaker in case values are equal.
                elif cur_q_val == best_action_score:
                    best_action = random.choice([best_action, action])
            return best_action

    def get_action(self, state, reward):
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
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        # can't learn from the first move, so let it be random.
        if self.prev_state is None:
            # update prev action to a random action and return this as the current action
            # also update the state to the current state.
            self.prev_state = state
            self.prev_action = random.choice(legalActions)
            return self.prev_action

        # We need to record our actions as tuples = (prev_state, prev_action, reward, cur_state)
        # and add these actions to our database

        db_tup = (self.prev_state, self.prev_action, reward, state)
        self.db.append(db_tup)

        if not legalActions:
            return action

        # Decide on action being exploration or exploitation
        # exploration
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        # exploitation
        else:
            action = self.getPolicy(state)

        # update previous state and action to be the ones being taken right now. then return the action
        self.prev_state = state
        self.prev_action = action
        return action

    def learn(self):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
        """
        "*** YOUR CODE HERE ***"
        # self.cur_learner(round,prev_state,prev_action, reward,new_state, too_slow)
        if len(self.db) < self.batch_size:
            return

        state_array = []
        reward_array = []

        samples = choices(self.db, k=self.batch_size)
        for tuple in samples:
            tup_prev_vector, tup_reward, tup_new_state, tup_best_action = tuple
            best_action_score = self.getValue(tup_new_state)
            target = tup_reward + self.discount * best_action_score

            state_array.append(tup_prev_vector.reshape(1, -1))
            reward_array.append(target)

        reward_array = np.array(reward_array)[:, np.newaxis, np.newaxis]
        state_array = np.array(state_array)
        self.network.fit(state_array, reward_array, epochs=1, verbose=0)
        # update epsilon to make it lower

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

