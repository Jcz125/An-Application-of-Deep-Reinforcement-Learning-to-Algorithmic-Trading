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
import yaml

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

###############################################################################
############################ Class DRLAgent BASE ##############################
###############################################################################

class DRLAgent:
    def __init__(self, observationSpace, actionSpace, configsFile):
        # Set variables from config file
        self.strategyName = self.__class__.__name__
        with open(configsFile, 'r') as yamlfile:
            configs = yaml.safe_load(yamlfile)
        self.model_params = configs["model"]
        self.numberOfNeurons = self.model_params["numberOfNeurons"]
        self.dropout = self.model_params["dropout"]
        self.gamma = self.model_params["gamma"]
        self.learningRate = self.model_params["learningRate"]
        self.targetNetworkUpdate = self.model_params["targetNetworkUpdate"]
        self.learningUpdatePeriod = self.model_params["learningUpdatePeriod"]
        self.experiencesRequired = self.model_params["experiencesRequired"]
        self.epsilonStart = self.model_params["epsilonStart"]
        self.epsilonEnd = self.model_params["epsilonEnd"]
        self.epsilonDecay = self.model_params["epsilonDecay"]
        self.capacity = self.model_params["capacity"]
        self.batchSize = self.model_params["batchSize"]
        self.L2Factor = self.model_params["L2Factor"]
        self.alpha = self.model_params["alpha"]
        self.filterOrder = self.model_params["filterOrder"]
        self.gradientClipping = self.model_params["gradientClipping"]
        self.rewardClipping = self.model_params["rewardClipping"]
        self.GPUNumber = self.model_params["GPUNumber"]
        self.ending_date = '2020-1-1' # self.configs['environment']['endingDate']
        # Check availability of CUDA for the hardware (CPU or GPU)
        self.device = torch.device('cuda:' + str(self.GPUNumber) if torch.cuda.is_available() else 'cpu')
        # Initialise the random function with a new random seed
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        # Set both the observation and action spaces
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        # Initialization of the iterations counter
        self.iterations = 0
        # Initialization of the tensorboard writer
        run_time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        self.writer = SummaryWriter('runs/' + run_time, comment=f'strategy: {self.strategyName}')
        self.writer.add_text(
            f"{self.strategyName} {run_time}", 
            f"Model-parameters: \n{str(self.model_params)}"
        )


    def getNormalizationCoefficients(self, tradingEnv):
        """
        GOAL: Retrieve the coefficients required for the normalization
              of input data.
        
        INPUTS: - tradingEnv: RL trading environement to process.
        
        OUTPUTS: - coefficients: Normalization coefficients.
        """
        # Retrieve the available trading data
        tradingData = tradingEnv.data
        closePrices = tradingData['Close'].tolist()
        lowPrices = tradingData['Low'].tolist()
        highPrices = tradingData['High'].tolist()
        volumes = tradingData['Volume'].tolist()
        # Retrieve the coefficients required for the normalization
        coefficients = []
        margin = 1
        # 1. Close price => returns (absolute) => maximum value (absolute)
        returns = [abs((closePrices[i] - closePrices[i - 1]) / closePrices[i - 1]) for i in range(1, len(closePrices))]
        coeffs = (0, np.max(returns) * margin)
        coefficients.append(coeffs)
        # 2. Low/High prices => Delta prices => maximum value
        deltaPrice = [abs(highPrices[i] - lowPrices[i]) for i in range(len(lowPrices))]
        coeffs = (0, np.max(deltaPrice) * margin)
        coefficients.append(coeffs)
        # 3. Close/Low/High prices => Close price position => no normalization required
        coeffs = (0, 1)
        coefficients.append(coeffs)
        # 4. Volumes => minimum and maximum values
        coeffs = (np.min(volumes) / margin, np.max(volumes) * margin)
        coefficients.append(coeffs)
        return coefficients
    

    def processState(self, state, coefficients):
        """
        GOAL: Process the RL state returned by the environment
              (appropriate format and normalization).

        INPUTS: - state: RL state returned by the environment.

        OUTPUTS: - state: Processed RL state.
        """
        # Normalization of the RL state
        closePrices = state[0]
        lowPrices = state[1]
        highPrices = state[2]
        volumes = state[3]
        # 1. Close price => returns => MinMax normalization
        returns = [(closePrices[i] - closePrices[i - 1]) / closePrices[i - 1] for i in range(1, len(closePrices))]
        if coefficients[0][0] != coefficients[0][1]:
            state[0] = [((x - coefficients[0][0]) / (coefficients[0][1] - coefficients[0][0])) for x in returns]
        else:
            state[0] = [0 for x in returns]
        # 2. Low/High prices => Delta prices => MinMax normalization
        deltaPrice = [abs(highPrices[i] - lowPrices[i]) for i in range(1, len(lowPrices))]
        if coefficients[1][0] != coefficients[1][1]:
            state[1] = [((x - coefficients[1][0]) / (coefficients[1][1] - coefficients[1][0])) for x in deltaPrice]
        else:
            state[1] = [0 for x in deltaPrice]
        # 3. Close/Low/High prices => Close price position => No normalization required
        closePricePosition = []
        for i in range(1, len(closePrices)):
            deltaPrice = abs(highPrices[i] - lowPrices[i])
            if deltaPrice != 0:
                item = abs(closePrices[i] - lowPrices[i]) / deltaPrice
            else:
                item = 0.5
            closePricePosition.append(item)
        if coefficients[2][0] != coefficients[2][1]:
            state[2] = [((x - coefficients[2][0]) / (coefficients[2][1] - coefficients[2][0])) for x in
                        closePricePosition]
        else:
            state[2] = [0.5 for x in closePricePosition]
        # 4. Volumes => MinMax normalization
        volumes = [volumes[i] for i in range(1, len(volumes))]
        if coefficients[3][0] != coefficients[3][1]:
            state[3] = [((x - coefficients[3][0]) / (coefficients[3][1] - coefficients[3][0])) for x in volumes]
        else:
            state[3] = [0 for x in volumes]
        ### turn context into returns for each of them and min max them
        for ii in range(4, len(state) - 1):
            context_series = state[ii]
            returns = [
                (context_series[i] - context_series[i - 1]) / context_series[i - 1] if context_series[i - 1] != 0 else 0
                for i in range(1, len(context_series))
            ]
            max_return = max(returns)
            state[ii] = [(x / (max_return)) if max_return != 0 else 0 for x in returns]
        # Process the state structure to obtain the appropriate format
        state = [item for sublist in state for item in sublist]
        return state


    def processReward(self, reward):
        """
        GOAL: Process the RL reward returned by the environment by clipping
              its value. Such technique has been shown to improve the stability
              the DQN algorithm.
        
        INPUTS: - reward: RL reward returned by the environment.
        
        OUTPUTS: - reward: Process RL reward.
        """
        return np.clip(reward, -self.rewardClipping, self.rewardClipping)


    def updateTargetNetwork(self):
        """
        GOAL: Taking into account the update frequency (parameter), update the
              target Deep Neural Network by copying the policy Deep Neural Network
              parameters (weights, bias, etc.).
        
        INPUTS: /
        
        OUTPUTS: /
        """
        # Check if an update is required (update frequency)
        if (self.iterations % self.targetNetworkUpdate == 0):
            # Transfer the DNN parameters (policy network -> target network)
            self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())


    def chooseActionEpsilonGreedy(self, state, previousAction):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed, following the 
              Epsilon Greedy exploration mechanism.
        
        INPUTS: - state: RL state returned by the environment.
                - previousAction: Previous RL action executed by the agent.
        
        OUTPUTS: - action: RL action chosen from the action space.
                 - Q: State-action value function associated.
                 - QValues: Array of all the Qvalues outputted by the
                            Deep Neural Network.
        """
        # EXPLOITATION -> RL policy
        if (random.random() > self.epsilonValue(self.iterations)):
            # Sticky action (RL generalization mechanism)
            if (random.random() > self.alpha):
                action, Q, QValues = self.chooseAction(state)
            else:
                action = previousAction
                Q = 0
                QValues = [0, 0]
        # EXPLORATION -> Random
        else:
            action = random.randrange(self.actionSpace)
            Q = 0
            QValues = [0, 0]
        # Increment the iterations counter (for Epsilon Greedy)
        self.iterations += 1
        return action, Q, QValues


    def plotTraining(self, score, marketSymbol, displayOption=DisplayOption()):
        """
        GOAL: Plot the training phase results
              (score, sum of rewards).
        
        INPUTS: - score: Array of total episode rewards.
                - marketSymbol: Stock market trading symbol.
        
        OUTPUTS: /
        """
        displayManager = DisplayManager(displayOptions=displayOption)
        ax1 = displayManager.add_subplot(111, ylabel='Total reward collected', xlabel='Episode')
        ax1.plot(score)
        displayManager.show(f"{str(marketSymbol)}_{self.strategyName}_Training_Results")


    def plotQValues(self, QValues0, QValues1, marketSymbol, displayOption=DisplayOption(), extraText=""):
        """
        Plot sequentially the Q values related to both actions.
        
        :param: - QValues0: Array of Q values linked to action 0.
                - QValues1: Array of Q values linked to action 1.
                - marketSymbol: Stock market trading symbol.
        
        :return: /
        """
        displayManager = DisplayManager(displayOptions=displayOption)
        ax1 = displayManager.add_subplot(111, ylabel='Q values', xlabel='Time')
        ax1.plot(QValues0)
        ax1.plot(QValues1)
        ax1.legend(['Short', 'Long'])
        displayManager.show(f"{str(marketSymbol)}__{self.strategyName}_{extraText}_QValues")


    def plotExpectedPerformance(self, trainingEnv, trainingParameters=[], iterations=10, 
                                trainingTestingPerformanceDisplayOption=DisplayOption(),
                                trainingTestingExpectedPerformanceDisplayOption=DisplayOption()):
        """
        GOAL: Plot the expected performance of the intelligent DRL trading agent.
        
        INPUTS: - trainingEnv: Training RL environment (known).
                - trainingParameters: Additional parameters associated
                                      with the training phase (e.g. the number
                                      of episodes). 
                - iterations: Number of training/testing iterations to compute
                              the expected performance.
        
        OUTPUTS: - trainingEnv: Training RL environment.
        """
        # Preprocessing of the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)
        # Save the initial Deep Neural Network weights
        initialWeights =  copy.deepcopy(self.policyNetwork.state_dict())
        # Initialization of some variables tracking both training and testing performances
        performanceTrain = np.zeros((trainingParameters[0], iterations))
        performanceTest = np.zeros((trainingParameters[0], iterations))
        # Initialization of the testing trading environment
        marketSymbol = trainingEnv.marketSymbol
        startingDate = trainingEnv.endingDate
        endingDate = '2020-1-1'
        money = trainingEnv.data['Money'][0]
        stateLength = trainingEnv.stateLength
        transactionCosts = trainingEnv.transactionCosts
        testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)
        # Print the hardware selected for the training of the Deep Neural Network (either CPU or GPU)
        print("Hardware selected for training: " + str(self.device))
        try:
            # Apply the training/testing procedure for the number of iterations specified
            for iteration in range(iterations):
                # Print the progression
                print(''.join(["Expected performance evaluation progression: ", str(iteration+1), "/", str(iterations)]))
                # Training phase for the number of episodes specified as parameter
                for episode in tqdm(range(trainingParameters[0])):
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
                        # Interact with the training environment until termination
                        while done == 0:
                            # Choose an action according to the RL policy and the current RL state
                            action, _, _ = self.chooseActionEpsilonGreedy(state, previousAction)
                            # Interact with the environment with the chosen action
                            nextState, reward, done, info = trainingEnvList[i].step(action)
                            # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                            reward = self.processReward(reward)
                            nextState = self.processState(nextState, coefficients)
                            self.replayMemory.push(state, action, reward, nextState, done)
                            # Trick for better exploration
                            otherAction = int(not bool(action))
                            otherReward = self.processReward(info['Reward'])
                            otherDone = info['Done']
                            otherNextState = self.processState(info['State'], coefficients)
                            self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)
                            # Execute the DQN learning procedure
                            stepsCounter += 1
                            if stepsCounter == self.learningUpdatePeriod:
                                self.learning()
                                stepsCounter = 0
                            # Update the RL state
                            state = nextState
                            previousAction = action
                    # Compute both training and testing  current performances
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performanceTrain[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performanceTrain[episode][iteration], episode)     
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performanceTest[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performanceTest[episode][iteration], episode)
                # Restore the initial state of the intelligent RL agent
                if iteration < (iterations-1):
                    trainingEnv.reset()
                    testingEnv.reset()
                    self.policyNetwork.load_state_dict(initialWeights)
                    self.targetNetwork.load_state_dict(initialWeights)
                    self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=self.learningRate, weight_decay=self.L2Factor)
                    self.replayMemory.reset()
                    self.iterations = 0
                    stepsCounter = 0
            
            iteration += 1
        except KeyboardInterrupt:
            print()
            print("WARNING: Expected performance evaluation prematurely interrupted...")
            print()
            self.policyNetwork.eval()
        # Compute the expected performance of the intelligent DRL trading agent
        expectedPerformanceTrain = []
        expectedPerformanceTest = []
        stdPerformanceTrain = []
        stdPerformanceTest = []
        for episode in range(trainingParameters[0]):
            expectedPerformanceTrain.append(np.mean(performanceTrain[episode][:iteration]))
            expectedPerformanceTest.append(np.mean(performanceTest[episode][:iteration]))
            stdPerformanceTrain.append(np.std(performanceTrain[episode][:iteration]))
            stdPerformanceTest.append(np.std(performanceTest[episode][:iteration]))
        expectedPerformanceTrain = np.array(expectedPerformanceTrain)
        expectedPerformanceTest = np.array(expectedPerformanceTest)
        stdPerformanceTrain = np.array(stdPerformanceTrain)
        stdPerformanceTest = np.array(stdPerformanceTest)
        # Plot each training/testing iteration performance of the intelligent DRL trading agent
        displayManager = DisplayManager(displayOptions=trainingTestingPerformanceDisplayOption)
        for i in range(iteration):
            ax = displayManager.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot([performanceTrain[e][i] for e in range(trainingParameters[0])])
            ax.plot([performanceTest[e][i] for e in range(trainingParameters[0])])
            ax.legend(["Training", "Testing"])
            displayManager.show(f"{str(marketSymbol)}_{self.strategyName}_Training_Testing_Performance_{str(i+1)}")
        # Plot the expected performance of the intelligent DRL trading agent
        displayManager = DisplayManager(displayOptions=trainingTestingExpectedPerformanceDisplayOption)
        ax = displayManager.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
        ax.plot(expectedPerformanceTrain)
        ax.plot(expectedPerformanceTest)
        ax.fill_between(range(len(expectedPerformanceTrain)), expectedPerformanceTrain-stdPerformanceTrain, expectedPerformanceTrain+stdPerformanceTrain, alpha=0.25)
        ax.fill_between(range(len(expectedPerformanceTest)), expectedPerformanceTest-stdPerformanceTest, expectedPerformanceTest+stdPerformanceTest, alpha=0.25)
        ax.legend(["Training", "Testing"])
        displayManager.show(f"{str(marketSymbol)}_{self.strategyName}_Training_Testing_Expected_Performance")
        # Closing of the tensorboard writer
        self.writer.close()
        return trainingEnv


    def saveModel(self, fileName):
        """
        GOAL: Save the RL policy, which is the policy Deep Neural Network.
        
        INPUTS: - fileName: Name of the file.
        
        OUTPUTS: /
        """
        torch.save(self.policyNetwork.state_dict(), fileName)


    def loadModel(self, fileName):
        """
        GOAL: Load a RL policy, which is the policy Deep Neural Network.
        
        INPUTS: - fileName: Name of the file.
        
        OUTPUTS: /
        """
        self.policyNetwork.load_state_dict(torch.load(fileName, map_location=self.device))
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())


    def plotEpsilonAnnealing(self, displayOption=DisplayOption()):
        """
        GOAL: Plot the annealing behaviour of the Epsilon variable
              (Epsilon-Greedy exploration technique).
        
        INPUTS: /
        
        OUTPUTS: /
        """
        displayManager = DisplayManager(displayOptions=displayOption)
        fig = displayManager.getFigure()
        plt.plot([self.epsilonValue(i) for i in range(10*self.epsilonDecay)])
        plt.xlabel("Iterations")
        plt.ylabel("Epsilon value")
        displayManager.show(f"EpsilonAnnealing_{self.strategyName}")


    def chooseAction(self, state):
        pass


    def learning(self, batchSize):
        pass


    def training(self, 
                 trainingEnv,
                 context={}, 
                 trainingParameters=[],
                 verbose=False, 
                 rendering=DisplayOption(), 
                 plotTraining=DisplayOption(), 
                 showPerformance=False,
                 interactiveTradingGraph=False):
        pass
        
    
    def learningBatch(self, batch_size):
        pass


    def trainingBatch(self, 
                      trainingEnv,
                      context={},
                      trainingParameters=[], 
                      batch_size=32,
                      verbose=False, 
                      rendering=DisplayOption(), 
                      plotTraining=DisplayOption(), 
                      showPerformance=False,
                      interactiveTradingGraph=False):
        pass


    def testing(self, 
                trainingEnv, 
                testingEnv, 
                rendering=DisplayOption(), 
                showPerformance=False, 
                interactiveTradingGraph=False):
        pass
