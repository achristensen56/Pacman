# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from PIL import ImageGrab
from sklearn_theano.feature_extraction import OverfeatTransformer
from skimage.measure import block_reduce
import random,util,math

class QLearningAgent(ReinforcementAgent):
            """
              Q-Learning Agent

              Functions you should fill in:
                - computeValueFromQValues
                - computeActionFromQValues
                - getQValue
                - getAction
                - update

              Instance variables you have access to
                - self.epsilon (exploration prob)
                - self.alpha (learning rate)
                - self.discount (discount rate)

              Functions you should use
                - self.getLegalActions(state)
                  which returns legal actions for a state
            """
            def __init__(self, **args):
                "You can initialize Q-values here..."
                ReinforcementAgent.__init__(self, **args)

                self.qValues = util.Counter()


            def getQValue(self, state, action):
                """
                  Returns Q(state,action)
                  Should return 0.0 if we have never seen a state
                  or the Q node value otherwise
                """
                return self.qValues[(state, action)]


            def computeValueFromQValues(self, state):
                """
                  Returns max_action Q(state,action)
                  where the max is over legal actions.  Note that if
                  there are no legal actions, which is the case at the
                  terminal state, you should return a value of 0.0.
                """
                bestAction = self.computeActionFromQValues(state)
                if bestAction == None:
                    return 0.0
                return self.getQValue(state, bestAction)

            def computeActionFromQValues(self, state):
                """
                  Compute the best action to take in a state.  Note that if there
                  are no legal actions, which is the case at the terminal state,
                  you should return None.
                """
                actions = self.getLegalActions(state)
                if len(actions) == 0:
                    return None
                qVals = [self.getQValue(state, a) for a in actions]
                bestActions = []
                bestVal = max(qVals)
                for i in range(len(actions)):
                    if qVals[i] == bestVal:
                        bestActions.append(actions[i])
                return random.choice(bestActions) #Break ties randomly

            def getAction(self, state):
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

                if len(legalActions) == 0:
                    return None
                useRandomAction = util.flipCoin(self.epsilon)
                if useRandomAction:
                    action = random.choice(legalActions)
                else:
                    action = self.computeActionFromQValues(state)

                return action

            def update(self, state, action, nextState, reward):
                """
                  The parent class calls this to observe a
                  state = action => nextState and reward transition.
                  You should do your Q-Value update here

                  NOTE: You should never call this function,
                  it will be called on your behalf
                """
                oldComponent = (1-self.alpha) * self.getQValue(state, action)
                nextValue = self.computeValueFromQValues(nextState)
                sample = reward + self.discount * nextValue
                newComponent = self.alpha * sample
                self.qValues[(state, action)] = oldComponent + newComponent


            def getPolicy(self, state):
                return self.computeActionFromQValues(state)

            def getValue(self, state):
                return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='SimpleExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        total = 0
        for feat in features:
            total += self.getWeights()[feat] * features[feat]
        return total

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        candidateQ = reward + self.discount * \
            self.computeValueFromQValues(nextState)
        currentQ = self.getQValue(state, action)
        difference = candidateQ - currentQ
        features = self.featExtractor.getFeatures(state, action)
        for feat in features:
            self.weights[feat] += self.alpha * difference * features[feat]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

import numpy as np
import tensorflow as tf

class NeuralNetAgent(PacmanQAgent):
  def __init__(self, **args):
    """Uses the SimpleExtractor feature extractor with a 2-layer network
    (affine-relu-affine-relu) with an L2 loss to predict Q values."""
    self.featExtractor = util.lookup("SimpleExtractor", globals())()
    PacmanQAgent.__init__(self, **args)
    self.sess = tf.Session()
    self.allFeats = list(self.featExtractor.getFeatures(None, None))
    numDims = len(self.allFeats)
    self.x = tf.placeholder("float", shape=[None, numDims])
    self.y_ = tf.placeholder("float", shape=[None, 1])
    self.W1 = tf.Variable(tf.truncated_normal([numDims, 5], stddev=0.01))
    self.b1 = tf.Variable(tf.constant(0.01, shape=[5]))
    self.W2 = tf.Variable(tf.truncated_normal([5, 1]))
    self.b2 = tf.Variable(tf.constant(0.01, shape=[1]))
    self.h = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)
    self.out = tf.nn.relu(tf.matmul(self.h, self.W2) + self.b2)
    self.l2_loss = tf.reduce_sum(tf.square(self.y_ - self.out))
    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.l2_loss)
    self.sess.run(tf.initialize_all_variables())

  def getQValue(self, state, action):
    """Performs a forward pass through the network to predict Q(state, action).
    """
    features = self.featExtractor.getFeatures(state, action)
    features_as_list = [features[feat] for feat in self.allFeats]
    features_vec = np.array([features_as_list])
    result = self.out.eval(feed_dict = {self.x: features_vec},
        session=self.sess)
    return result[0][0]


  def update(self, state, action, nextState, reward):
    """Performs a training step on the network."""
    candidateQ = reward + self.discount * \
        self.computeValueFromQValues(nextState)
    candidateQ_vec = np.array([[candidateQ]])
    features = self.featExtractor.getFeatures(state, action)
    features_as_list = [features[feat] for feat in self.allFeats]
    features_vec = np.array([features_as_list])
    self.train_step.run(feed_dict={self.x: features_vec,
      self.y_: candidateQ_vec}, session=self.sess)

class ConvQAgent(PacmanQAgent):
  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

        self.tf = OverfeatTransformer(output_layers = [-1], force_reshape = False)
        self.weights = util.Counter()

  def featureExtractor(self):

    screen = np.array(ImageGrab.grab(bbox = (50, 120, 1250, 650)))
    small_screen = block_reduce(screen, (530/224, 1200/224, 1), np.max)
    features = np.array(self.tf.transform(small_screen[:,:,0:3])).flatten()

    feat = dict(zip(range(2000), features))

    return  feat
  def getQValue(self, state, action):
    features = self.featureExtractor()
    total = 0
    for feat in features:
      total += feat * self.weights[feat]
    return total

  def update(self, state, action, nextState, reward):
      candidateQ = reward + self.discount * \
          self.computeValueFromQValues(nextState)
      currentQ = self.getQValue(state, action)
      features = self.featureExtractor()
      difference = candidateQ - currentQ
      for feat in features:
        self.weights[feat] += self.alpha * difference * feat




