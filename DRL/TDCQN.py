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
import os
import random
import copy
import datetime

import numpy as np

from collections import deque

import yaml
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from DRL.DRLAgent import DRLAgent
from Misc.displayManager import DisplayManager, DisplayOption

from DataProcessing.tradingPerformance import PerformanceEstimator
from DataProcessing.dataAugmentation import DataAugmentation
from Environment.tradingEnv import TradingEnv
from Misc.displayManager import *
from DRL.TDQNBase import TDQNBase

###############################################################################
############################## Class ReplayMemoryCNN ##########################
###############################################################################

class ReplayMemoryCNN:
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
        self.memory = list()
        self.capacity = capacity


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


    def sample(self, batchSize, timesteps=1):
        """
        GOAL: Sample a batch of experiences from the replay memory.
        INPUTS: - batchSize: Size of the batch to sample.
        OUTPUTS: - state: RL states of the experience batch sampled.
                 - action: RL actions of the experience batch sampled.
                 - reward: RL rewards of the experience batch sampled.
                 - nextState: RL next states of the experience batch sampled.
                 - done: RL termination signals of the experience batch sampled.
        """
        if timesteps < 2:
            state, action, reward, nextState, done = zip(*random.sample(self.memory, batchSize))
            return state, action, reward, nextState, done
        idx = random.sample(range(0, len(self.memory) - timesteps + 1), batchSize)
        state, action, reward, nextState, done = [], [], [], [], []
        for i in idx:
            sampled = self.memory[i:i + timesteps]
            s, a, r, n_s, d = zip(*sampled)
            state.append(s)
            action.append(a[-1])
            reward.append(r[-1])
            nextState.append(n_s[1:])
            done.append(d[-1])
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
        self.memory = list()

###############################################################################
################################### Class DQN #################################
###############################################################################


class LinearBlock(nn.Module):
    def __init__(self, numberOfInputs, numberOfOutputs, dropout, **kwargs):
        super().__init__()

        # Definition of some Fully Connected layers
        self.block = nn.Sequential(
            nn.Linear(numberOfInputs, numberOfOutputs), 
            nn.ReLU(),
            nn.BatchNorm1d(numberOfOutputs), 
            nn.Dropout(dropout)
        )
        # Xavier initialization for the entire neural network
        torch.nn.init.xavier_uniform_(self.block[0].weight)

    def forward(self, x):
        return self.block(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_input, n_output, dropout, **kwargs):
        super().__init__()
        self.proj = nn.Linear(n_input, n_output)
        self.block = nn.TransformerEncoderLayer(
            n_output,
            4,
            dim_feedforward=512,
            dropout=dropout
        )

    def forward(self, x):
        """
        x: torch.tensor of shape [N, T, E]
        return: torch.tensor of shape [N, T, E]
        """
        reshaped = x.permute(1, 0, 2)  # [T, N, E]
        att = self.block(self.proj(reshaped))
        return att.permute(1, 0, 2)  # [N, T, E]


class ConvBlock(nn.Module):
    def __init__(self,
                 numberOfInputs,
                 numberOfOutputs,
                 dropout,
                 kernel_size=2,
                 stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                numberOfInputs,
                numberOfOutputs,
                kernel_size=kernel_size,
                stride=stride
            ), 
            nn.LeakyReLU(), 
            nn.BatchNorm1d(numberOfOutputs),
            nn.Dropout(dropout)
        )
        torch.nn.init.xavier_uniform_(self.block[0].weight)

    def forward(self, x):
        return self.block(x)


class DCQN(nn.Module):
    """
    GOAL: Implementing the Deep Neural Network of the DQN Reinforcement
          Learning algorithm.
    """
    def __init__(self,
                 numberOfInputs,
                 numberOfOutputs,
                 numberOfNeurons,
                 numberOfLayers=5,
                 blockType='linear',
                 dropout=0.5):
        """
        GOAL: Defining and initializing the Deep Neural Network of the
              DQN Reinforcement Learning algorithm.
        """
        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super().__init__()
        self.blockType = blockType
        conv_block_args = { 'kernel_size': [2, 2, 2], 'stride': [1, 1, 1] }
        if blockType == 'LINEAR': block = LinearBlock
        elif blockType == 'ATTENTION': block = AttentionBlock
        elif blockType == 'CONV': block = ConvBlock
        input_block = block(numberOfInputs, numberOfNeurons, dropout=dropout)
        hidden_blocks = [
            block(numberOfNeurons, numberOfNeurons, dropout=dropout, kernel_size=conv_block_args['kernel_size'][i], stride=conv_block_args['stride'][i])
                if blockType == 'CONV' else block(numberOfNeurons, numberOfNeurons, dropout=dropout)
            for i in range(numberOfLayers - 2)
        ]
        self.hidden_layers = nn.Sequential(input_block, *hidden_blocks)
        self.out_layer = nn.Linear(numberOfNeurons, numberOfOutputs)
        self.inter_layer = nn.Linear(4, 2)

    def forward(self, x):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.
        """
        if self.blockType == 'CONV': x = x.permute(1, 2, 0)
        x = self.hidden_layers(x)
        if len(x.shape) > 2:
            x = x[:, -1, :]
        return self.out_layer(x)


###############################################################################
################################ Class TDRQN ###################################
###############################################################################

class TDCQN(DRLAgent):
    """
    GOAL: Implementing an intelligent trading agent based on the DQN
          Reinforcement Learning algorithm.
    """
    def __init__(self, observationSpace, actionSpace, configsFile='./Configurations/hyperparameters-tdcqn.yml', blockType='LINEAR'):
        """
        GOAL: Initializing the RL agent based on the DQN Reinforcement Learning
              algorithm, by setting up the DQN algorithm parameters as well as
              the DQN Deep Neural Network.

        INPUTS: - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - run_config: Path to configuration file containing all env and model params
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
        self.blockType = blockType
        self.strategyName += f'_{self.blockType}'
        self.timesteps = self.model_params['timesteps'] if self.blockType != 'LINEAR' else 1
        self.numberOfLayers = self.model_params['numberOfLayers']
        # Set the Experience Replay mechnism
        self.replayMemory = ReplayMemoryCNN(self.capacity)
        # Set the Epsilon-Greedy exploration technique
        self.epsilonValue = lambda iteration: self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * math.exp(-1 * iteration / self.epsilonDecay)
        # Set the two Deep Neural Networks of the DQN algorithm (policy and target)
        self.policyNetwork = DCQN(observationSpace, 
                                  actionSpace, 
                                  self.numberOfNeurons, 
                                  numberOfLayers=self.numberOfLayers, 
                                  blockType=self.blockType, 
                                  dropout=self.dropout).to(self.device)
        self.targetNetwork = DCQN(observationSpace, 
                                  actionSpace, 
                                  self.numberOfNeurons, 
                                  numberOfLayers=self.numberOfLayers, 
                                  blockType=self.blockType, 
                                  dropout=self.dropout).to(self.device)
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
        self.policyNetwork.eval()
        self.targetNetwork.eval()
        # Set the Deep Learning optimizer
        self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=self.learningRate, weight_decay=self.L2Factor)
    

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
        state_space = len(state[-1])
        state = [[0] * state_space if s is None else s for s in state]
        if self.timesteps == 1:
            state = state[0]
        with torch.no_grad():
            tensorState = torch.tensor(state).float().to(self.device).unsqueeze(0)
            QValues = self.policyNetwork(tensorState).squeeze(0)
            Q, action = QValues.max(0)
            action = action.item()
            Q = Q.item()
            QValues = QValues.cpu().numpy()
            return action, Q, QValues


    def learning(self, batchSize):
        """
        GOAL: Sample a batch of past experiences and learn from it
              by updating the Reinforcement Learning policy.
        INPUTS: batchSize: Size of the batch to sample from the replay memory.
        OUTPUTS: /
        """
        # Check that the replay memory is filled enough
        if (len(self.replayMemory) - self.timesteps + 1 >= batchSize):
            # Set the Deep Neural Network in training mode
            self.policyNetwork.train()
            # Sample a batch of experiences from the replay memory
            state, action, reward, nextState, done = self.replayMemory.sample(batchSize, timesteps=self.timesteps)
            # Initialization of Pytorch tensors for the RL experience elements
            state = torch.tensor(state, dtype=torch.float, device=self.device)  # [N, T, E]
            action = torch.tensor(action).long().to(self.device)  # [N,]
            reward = torch.tensor(reward).float().to(self.device)  # [N,]
            nextState = torch.tensor(nextState).float().to(self.device)  # [N, T-1, E]
            done = torch.tensor(done).float().to(self.device)  # [N,]
            # Compute the current Q values returned by the policy network
            currentQValues = self.policyNetwork(state).gather(-1, action.unsqueeze(-1)).squeeze(-1)  # [N,]
            # Compute the next Q values returned by the target network
            with torch.no_grad():
                nextActions = torch.max(self.policyNetwork(nextState), -1)[1]
                nextQValues = self.targetNetwork(nextState).gather(-1, nextActions.unsqueeze(-1)).squeeze(-1)
                expectedQValues = reward + self.gamma * nextQValues * (1 - done)
            # Compute the Huber loss
            loss = F.smooth_l1_loss(currentQValues, expectedQValues)
            # Computation of the gradients
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.gradientClipping)
            # Perform the Deep Neural Network optimization
            self.optimizer.step()
            # If required, update the target deep neural network (update frequency)
            self.updateTargetNetwork()
            # Set back the Deep Neural Network in evaluation mode
            self.policyNetwork.eval()


    def training(self, 
                 trainingEnv,
                 context, 
                 trainingParameters=[],
                 verbose=False, 
                 rendering=DisplayOption(), 
                 plotTraining=DisplayOption(), 
                 showPerformance=False,
                 interactiveTradingGraph=False):
        """
        GOAL: Train the RL trading agent by interacting with its trading environment.

        INPUTS: - trainingEnv: Training RL environment (known).
                - trainingParameters: Additional parameters associated
                                      with the training phase (e.g. the number
                                      of episodes).
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the training environment rendering.
                - plotTraining: Enable the plotting of the training results.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.

        OUTPUTS: - trainingEnv: Training RL environment.
        """

        """
        # Compute and plot the expected performance of the trading policy
        trainingEnv = self.plotExpectedPerformance(trainingEnv, trainingParameters, iterations=50)
        return trainingEnv
        """
        verbose = verbose and not interactiveTradingGraph
        # Apply data augmentation techniques to improve the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)
        interactiveDisplayManager = DisplayManager(displayOptions=DisplayOption(False, False, True, rendering.recordVideo), 
                                                   figsize=default_fig_size) if interactiveTradingGraph or rendering.recordVideo else None

        # Initialization of some variables tracking the training and testing performances
        if plotTraining:
            # Training performance
            performanceTrain = []
            score = np.zeros((len(trainingEnvList), trainingParameters[0]))
            # Testing performance
            marketSymbol = trainingEnv.marketSymbol
            startingDate = trainingEnv.endingDate
            endingDate = self.ending_date
            money = trainingEnv.data['Money'][0]
            stateLength = trainingEnv.stateLength
            transactionCosts = trainingEnv.transactionCosts
            testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, context, stateLength, transactionCosts)
            performanceTest = []

        try:
            # If required, print the training progression
            if verbose:
                print("Training progression (hardware selected => " + str(self.device) + "):")
            # Training phase for the number of episodes specified as parameter
            for episode in tqdm(range(trainingParameters[0]), disable=not (verbose)):
                # For each episode, train on the entire set of training environments
                for i in range(len(trainingEnvList)):
                    # Set the initial RL variables
                    coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                    trainingEnvList[i].reset()
                    startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                    trainingEnvList[i].setStartingPoint(startingPoint)
                    state = self.processState(trainingEnvList[i].state, coefficients)
                    previousAction = 0
                    done = 0
                    stepsCounter = 0
                    # Set the performance tracking veriables
                    if plotTraining:
                        totalReward = 0
                    running_states = [None] * (self.timesteps - 1) + [state]
                    # Interact with the training environment until termination
                    while done == 0:
                        # Choose an action according to the RL policy and the current RL state
                        action, _, _ = self.chooseActionEpsilonGreedy(running_states, previousAction)
                        # Interact with the environment with the chosen action
                        nextState, reward, done, info = trainingEnvList[i].step(action)
                        # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                        reward = self.processReward(reward)
                        nextState = self.processState(nextState, coefficients)
                        running_states.pop(0)
                        running_states.append(nextState)
                        self.replayMemory.push(state, action, reward, nextState, done)
                        # Trick for better exploration
                        otherAction = int(not bool(action))
                        otherReward = self.processReward(info['Reward'])
                        otherNextState = self.processState(info['State'], coefficients)
                        otherDone = info['Done']
                        self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)
                        # Execute the DQN learning procedure
                        stepsCounter += 1
                        if stepsCounter == self.learningUpdatePeriod:
                            self.learning(self.batchSize)
                            stepsCounter = 0
                        # Update the RL state
                        state = nextState
                        previousAction = action
                        # Continuous tracking of the training performance
                        if plotTraining:
                            totalReward += reward
                        if interactiveDisplayManager:
                            trainingEnvList[i].render(_displayManager=interactiveDisplayManager)
                    # Store the current training results
                    if plotTraining:
                        score[i][episode] = totalReward
                    
                # Compute the current performance on both the training and testing sets
                if plotTraining:
                    # Training set performance
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTrain.append(performance)
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performance, episode)
                    trainingEnv.reset()
                    # Testing set performance
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTest.append(performance)
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performance, episode)
                    testingEnv.reset()
        except KeyboardInterrupt:
            print()
            print("WARNING: Training prematurely interrupted...")
            print()
            self.policyNetwork.eval()

        # Assess the algorithm performance on the training trading environment
        trainingEnv = self.testing(trainingEnv, trainingEnv)
        # If required, show the rendering of the trading environment
        if rendering:
            rendering.recordVideo = False
            trainingEnv.render(displayOptions=rendering, extraText="Training")
        # If required, plot the training results
        if plotTraining:
            displayManager = DisplayManager(displayOptions=plotTraining)
            ax = displayManager.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot(performanceTrain)
            ax.plot(performanceTest)
            ax.legend(["Training", "Testing"])
            displayManager.show(f"{str(marketSymbol)}_TrainingTestingPerformance")
            for i in range(len(trainingEnvList)):
                self.plotTraining(score[i][:episode], marketSymbol, displayOption=plotTraining)
        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance(f'{self.strategyName} (Training)')
        # Closing of the tensorboard writer
        self.writer.close()
        return trainingEnv


    def testing(self, 
                trainingEnv, 
                testingEnv, 
                rendering=DisplayOption(), 
                showPerformance=False, 
                interactiveTradingGraph=False):
        """
        GOAL: Test the RL agent trading policy on a new trading environment
              in order to assess the trading strategy performance.

        INPUTS: - trainingEnv: Training RL environment (known).
                - testingEnv: Unknown trading RL environment.
                - rendering: Enable the trading environment rendering.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.

        OUTPUTS: - testingEnv: Trading environment backtested.
        """
        # Apply data augmentation techniques to process the testing set
        dataAugmentation = DataAugmentation()
        testingEnvSmoothed = dataAugmentation.lowPassFilter(testingEnv, self.filterOrder)
        trainingEnv = dataAugmentation.lowPassFilter(trainingEnv, self.filterOrder)
        # Initialization of some RL variables
        coefficients = self.getNormalizationCoefficients(trainingEnv)
        state = self.processState(testingEnvSmoothed.reset(), coefficients)
        testingEnv.reset()
        QValues0 = []
        QValues1 = []
        done = 0
        running_states = [None] * (self.timesteps - 1) + [state]
        interactiveDisplayManager = DisplayManager(displayOptions=DisplayOption(False, False, True, rendering.recordVideo), 
                                                   figsize=default_fig_size) if interactiveTradingGraph or rendering.recordVideo else None
        # Interact with the environment until the episode termination
        while done == 0:
            # Choose an action according to the RL policy and the current RL state
            action, _, QValues = self.chooseAction(running_states)
            # Interact with the environment with the chosen action
            nextState, _, done, _ = testingEnvSmoothed.step(action)
            testingEnv.step(action)
            # Update the new state
            state = self.processState(nextState, coefficients)
            running_states.pop(0)
            running_states.append(state)
            # Storing of the Q values
            QValues0.append(QValues[0])
            QValues1.append(QValues[1])
            if interactiveDisplayManager:
                testingEnv.render(_displayManager=interactiveDisplayManager)
        # If required, show the rendering of the trading environment
        if rendering:
            testingEnv.render(displayOptions=rendering)
            self.plotQValues(QValues0, QValues1, testingEnv.marketSymbol, displayOption=rendering, extraText=self.strategyName)
        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance(f'{self.strategyName} (Testing)')
        return testingEnv