# coding=utf-8

"""
Goal: Implementing a custom enhanced version of the DQN algorithm specialized
      to algorithmic trading.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import math
import random
import copy
import datetime

import numpy as np

from collections import deque
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from DataProcessing.tradingPerformance import PerformanceEstimator
from DataProcessing.dataAugmentation import DataAugmentation
from Environment.tradingEnv import TradingEnv
from Misc.displayManager import *
from DRL.DRLAgent import *


###############################################################################
############################### Class ReplayMemory ############################
###############################################################################


class ReplayMemory:
    """
    GOAL: Implementing the replay memory required for the Experience Replay
          mechanism of the DQN Reinforcement Learning algorithm.
    
    VARIABLES:  - memory: Data structure storing the experiences.
                                
    METHODS:    - __init__: Initialization of the memory data structure.
                - push: Insert a new experience into the replay memory.
                - sample: Sample a batch of experiences from the replay memory.
                - __len__: Return the length of the replay memory.
                - reset: Reset the replay memory.
    """
    def __init__(self, capacity):
        """
        GOAL: Initializating the replay memory data structure.
        
        INPUTS: - capacity: Capacity of the data structure, specifying the
                            maximum number of experiences to be stored
                            simultaneously.
        
        OUTPUTS: /
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    

    def push(self, state, action, reward, nextState, done):
        """
        GOAL: Insert a new experience into the replay memory. An experience
              is composed of a state, an action, a reward, a next state and
              a termination signal.
        
        INPUTS: - state: RL state of the experience to be stored.
                - action: RL action of the experience to be stored.
                - reward: RL reward of the experience to be stored.
                - nextState: RL next state of the experience to be stored.
                - done: RL termination signal of the experience to be stored.
        
        OUTPUTS: /
        """
        self.memory.append((state, action, reward, nextState, done))


    def sample(self, batchSize):
        """
        GOAL: Sample a batch of experiences from the replay memory.
        
        INPUTS: - batchSize: Size of the batch to sample.
        
        OUTPUTS: - state: RL states of the experience batch sampled.
                 - action: RL actions of the experience batch sampled.
                 - reward: RL rewards of the experience batch sampled.
                 - nextState: RL next states of the experience batch sampled.
                 - done: RL termination signals of the experience batch sampled.
        """
        state, action, reward, nextState, done = zip(*random.sample(self.memory, batchSize))
        return state, action, reward, nextState, done


    def __len__(self):
        """
        GOAL: Return the capicity of the replay memory, which is the maximum number of
              experiences which can be simultaneously stored in the replay memory.
        
        INPUTS: /
        
        OUTPUTS: - length: Capacity of the replay memory.
        """
        return len(self.memory)


    def reset(self):
        """
        GOAL: Reset (empty) the replay memory.
        
        INPUTS: /
        
        OUTPUTS: /
        """
        self.memory = deque(maxlen=self.capacity)


###############################################################################
############################## Class DQN BASE #################################
###############################################################################

class TDQNBase(DRLAgent):
    def __init__(self, observationSpace, actionSpace, configsFile='./Configurations/hyperparameters-tdqn.yml'):
        """
        GOAL: Initializing the RL agent based on the DQN Reinforcement Learning
              algorithm, by setting up the DQN algorithm parameters as well as
              the DQN Deep Neural Network.

        INPUTS: - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - configs: Path to configuration file containing all env and model params
                - Other inputs specified by configuration file:
                - numberOfNeurons: Number of neurons per layer in the Deep Neural Network.
                - dropout: Droupout probability value (handling of overfitting).
                - gamma: Discount factor of the DQN algorithm.
                - learningRate: Learning rate of the ADAM optimizer.
                - targetNetworkUpdate: Update frequency of the target network.
                - epsilonStart: Initial (maximum) value of Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - epsilonEnd: Final (minimum) value of Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - epsilonDecay: Decay factor (exponential) of Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - capacity: Capacity of the Experience Replay memory.
                - batchSize: Size of the batch to sample from the replay memory.

        OUTPUTS: /
        """
        super().__init__(observationSpace, actionSpace, configsFile)
        # Set the Experience Replay mechnism
        self.replayMemory = ReplayMemory(self.capacity)
        # Set the Epsilon-Greedy exploration technique
        self.epsilonValue = lambda iteration: self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * math.exp(-1 * iteration / self.epsilonDecay)
    

    def chooseAction(self, state):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed.
        
        INPUTS: - state: RL state returned by the environment.
        
        OUTPUTS: - action: RL action chosen from the action space.
                 - Q: State-action value function associated.
                 - QValues: Array of all the Qvalues outputted by the
                            Deep Neural Network.
        """
        # Choose the best action based on the RL policy
        with torch.no_grad():
            tensorState = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            QValues = self.policyNetwork(tensorState)
            QValues = (QValues[0] if (isinstance(QValues, tuple)) else QValues).squeeze(0)
            Q, action = QValues.max(0)
            action = action.item()
            Q = Q.item()
            QValues = QValues.cpu().numpy()
            return action, Q, QValues
