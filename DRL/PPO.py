# coding=utf-8

"""
Goal: Implementing a custom version of the PPO algorithm specialized
      to algorithmic trading.
Authors: Alessandro Pavesi
Institution: University of Bologna
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import random
import copy
import datetime

import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from DataProcessing.tradingPerformance import PerformanceEstimator
from DataProcessing.dataAugmentation import DataAugmentation
from Environment.tradingEnv import TradingEnv
from Misc.displayManager import *
from DRL.DRLAgent import DRLAgent


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


    def length(self):
        assert len(self.actions) == len(self.states) == len(self.logprobs) == len(self.rewards) == len(self.is_terminals)
        return len(self.actions)


class ActorCritic(nn.Module):
    def __init__(self, 
                 numberOfInputs, 
                 numberOfOutputs, 
                 numberOfNeurons, 
                 numberOfLayers=1, 
                 blockType=''):
        super(ActorCritic, self).__init__()
        self.blockType = blockType
        if blockType == 'LSTM':
            self.rnn_block = nn.LSTM(
                input_size=numberOfInputs,
                hidden_size=numberOfInputs,
                num_layers=numberOfLayers,
                batch_first=True,
            )
        elif blockType == 'GRU':
            self.rnn_block = nn.GRU(
                input_size=numberOfInputs,
                hidden_size=numberOfInputs,
                num_layers=numberOfLayers,
                batch_first=True,
            )
        else:
            self.rnn_block = nn.Identity()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(numberOfInputs, numberOfNeurons),
            nn.LayerNorm(numberOfNeurons),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(numberOfNeurons, numberOfNeurons),
            nn.LayerNorm(numberOfNeurons),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(numberOfNeurons, numberOfNeurons),
            nn.LayerNorm(numberOfNeurons),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(numberOfNeurons, numberOfOutputs),
            nn.Softmax(dim=-1),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(numberOfInputs, numberOfNeurons),
            nn.LayerNorm(numberOfNeurons),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(numberOfNeurons, numberOfNeurons),
            nn.LayerNorm(numberOfNeurons),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(numberOfNeurons, numberOfNeurons),
            nn.LayerNorm(numberOfNeurons),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(numberOfNeurons, 1),
        )

    def forward(self):
        raise NotImplementedError

    def actor_part(self, state):
        out = state
        if self.blockType == "LSTM" or self.blockType == "GRU":
            out, _ = self.rnn_block(out.unsqueeze(0))
        action_probs = self.actor(out)
        dist = Categorical(action_probs)
        return dist

    def act(self, state):
        dist = self.actor_part(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        dist = self.actor_part(state)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


###############################################################################
################################ Class PPO ###################################
###############################################################################

class PPO(DRLAgent):
    """
    GOAL: Implementing an intelligent trading agent based on the PPO
          Reinforcement Learning algorithm.

    VARIABLES:  - device: Hardware specification (CPU or GPU).
                - gamma: Discount factor of the PPO algorithm.
                - learningRate: Learning rate of the ADAM optimizer.
                - capacity: Capacity of the experience replay memory.
                - batchSize: Size of the batch to sample from the replay memory.
                - targetNetworkUpdate: Frequency at which the target neural
                                       network is updated.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - policyNetwork: Deep Neural Network representing the RL policy.
                - targetNetwork: Deep Neural Network representing a target
                                 for the policy Deep Neural Network.
                - optimizer: Deep Neural Network optimizer (ADAM).
                - replayMemory: Experience replay memory.
                - epsilonValue: Value of the Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - iterations: Counter of the number of iterations.

    METHODS:    - __init__: Initialization of the RL trading agent, by setting up
                            many variables and parameters.
                - getNormalizationCoefficients: Retrieve the coefficients required
                                                for the normalization of input data.
                - processState: Process the RL state received.
                - processReward: Clipping of the RL reward received.
                - updateTargetNetwork: Update the target network, by transfering
                                       the policy network parameters.
                - chooseAction: Choose a valid action based on the current state
                                observed, according to the RL policy learned.
                - chooseActionEpsilonGreedy: Choose a valid action based on the
                                             current state observed, according to
                                             the RL policy learned, following the
                                             Epsilon Greedy exploration mechanism.
                - learn: Sample a batch of experiences and learn from that info.
                - training: Train the trading PPO agent by interacting with its
                            trading environment.
                - testing: Test the PPO agent trading policy on a new trading environment.
                - plotExpectedPerformance: Plot the expected performance of the intelligent
                                   DRL trading agent.
                - saveModel: Save the RL policy model.
                - loadModel: Load the RL policy model.
                - plotTraining: Plot the training results (score evolution, etc.).
                - plotEpsilonAnnealing: Plot the annealing behaviour of the Epsilon
                                     (Epsilon-Greedy exploration technique).
    """

    def __init__(self, observationSpace, actionSpace, configsFile='./Configurations/hyperparameters-ppo.yml', blockType='DEFAULT'):
        """
        GOAL: Initializing the RL agent based on the PPO Reinforcement Learning
              algorithm, by setting up the PPO algorithm parameters as well as
              the PPO Deep Neural Network.

        INPUTS: - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - numberOfNeurons: Number of neurons per layer in the Deep Neural Network.
                - gamma: Discount factor of the PPO algorithm.
                - learningRateActor: Learning rate of the ADAM optimizer for Actor.
                - learningRateCritic: Learning rate of the ADAM optimizer for Critic.

        OUTPUTS: /
        """
        super().__init__(observationSpace, actionSpace, configsFile)
        self.blockType = blockType
        self.strategyName += f'_{self.blockType}'
        self.numberOfLayers = self.model_params['numberOfLayers']
        # Set the general parameters of the PPO algorithm
        self.learningRateActor = self.model_params['learningRateActor']
        self.learningRateCritic = self.model_params['learningRateCritic']
        self.eps_clip = self.model_params['eps_clip']
        self.K_epochs = self.model_params['K_epochs']
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(observationSpace, 
                                  actionSpace, 
                                  self.numberOfNeurons, 
                                  numberOfLayers=self.numberOfLayers, 
                                  blockType=self.blockType).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.learningRateActor},
            {'params': self.policy.critic.parameters(), 'lr': self.learningRateCritic}
        ])
        self.policy_old = ActorCritic(observationSpace, 
                                      actionSpace, 
                                      self.numberOfNeurons,
                                      numberOfLayers=self.numberOfLayers, 
                                      blockType=self.blockType).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
    

    def chooseAction(self, state, append=True):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed.

        INPUTS: - state: RL state returned by the environment.
                - append: boolean defining if append or not the data to the buffer

        OUTPUTS: - action: RL action chosen from the action space.
                 - Q: State-action value function associated.
                 - QValues: Array of all the Qvalues outputted by the
                            Deep Neural Network.
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)
        if append:
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
        return action.item()


    def learning(self):
        #print(self.buffer.length())
        # Set the Deep Neural Network in training mode
        self.policy.train()
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # Final loss of clipped objective PPO (L[CLIP+VS+S])
            # loss = -LCLIP + C1*VF - C2*S
            # LOSS = PPOObjective + coef1*SquaredErrorLoss + coef2*EntropyBonus
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()
        # Set back the Deep Neural Network in evaluation mode
        self.policy.eval()

    def learningBatch(self, batch_size):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        old_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(self.device)
        # Set the Deep Neural Network in training mode
        self.policy.train()
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Calculate advantages with TD(0)
            advantages = []
            for _ in range(batch_size):
                reward = 0
                for t in range(len(state_values)-1):
                    rew = old_rewards[t]
                    is_term = old_terminals[t]
                    td = rew + self.gamma * state_values[t+1] * (1 - int(is_term)) - state_values[t]
                    reward += td**(batch_size-t+1)
                advantages.append(reward)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
            # Normalize advantages like rewards before
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # Final loss of clipped objective PPO (L[CLIP+VS+S])
            # loss = -LCLIP + C1*VF - C2*S
            # LOSS = PPOObjective + coef1*SquaredErrorLoss + coef2*EntropyBonus
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()
        # Set back the Deep Neural Network in evaluation mode
        self.policy.eval()

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
        interactiveDisplayManager = DisplayManager(displayOptions=DisplayOption(False, False, True, rendering.recordVideo), 
                                    figsize=default_fig_size) if interactiveTradingGraph or rendering.recordVideo else None
        # Apply data augmentation techniques to improve the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)
        # Initialization of some variables tracking the training and testing performances
        if plotTraining:
            # Training performance
            performanceTrain = []
            score = np.zeros((len(trainingEnvList), trainingParameters[0]))
            # Testing performance
            marketSymbol = trainingEnv.marketSymbol
            startingDate = trainingEnv.endingDate
            endingDate = '2020-1-1'
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
                    # Seems that if the starting point is the last but one the training fails
                    # because the batch is .size([])
                    # Moreover the last but one index is usually returned (How? there's a random!) so
                    # Check the startingPoint to not be the last but one
                    while startingPoint+1 == len(trainingEnvList[i].data.index):
                        startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                    trainingEnvList[i].setStartingPoint(startingPoint)
                    state = self.processState(trainingEnvList[i].state, coefficients)
                    done = 0
                    # Set the performance tracking veriables
                    if plotTraining:
                        totalReward = 0
                    # Interact with the training environment until termination
                    while done == 0:
                        # Choose an action according to the RL policy and the current RL state
                        action = self.chooseAction(state)
                        # Interact with the environment with the chosen action
                        nextState, reward, done, info = trainingEnvList[i].step(action)
                        # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                        reward = self.processReward(reward)
                        nextState = self.processState(nextState, coefficients)
                        self.buffer.rewards.append(reward)
                        self.buffer.is_terminals.append(done)
                        # Update the RL state
                        state = nextState
                        # Continuous tracking of the training performance
                        if plotTraining:
                            totalReward += reward
                        if interactiveDisplayManager:
                            trainingEnvList[i].render(_displayManager=interactiveDisplayManager)
                    # Execute the learning procedure
                    self.learning()
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
            self.policy.eval()

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
                self.plotTraining(score[i], marketSymbol, displayOption=plotTraining)
        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance(f'{self.strategyName} (Training)')
        # Closing of the tensorboard writer
        self.writer.close()
        return trainingEnv

    def trainingBatch(self, 
                      trainingEnv, 
                      context,
                      trainingParameters=[], 
                      batch_size=32,
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
        verbose = verbose and not interactiveTradingGraph
        # Apply data augmentation techniques to improve the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)
        interactiveDisplayManager = DisplayManager(displayOptions=DisplayOption(False, False, True, rendering.recordVideo), 
                                                   figsize=default_fig_size) if interactiveTradingGraph or rendering.recordVideo else None
        stepsCounter = 0
        # Initialization of some variables tracking the training and testing performances
        if plotTraining:
            # Training performance
            performanceTrain = []
            score = np.zeros((len(trainingEnvList), trainingParameters[0]))
            # Testing performance
            marketSymbol = trainingEnv.marketSymbol
            startingDate = trainingEnv.startingDate
            endingDate = trainingEnv.endingDate
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
                    done = 0
                    # Set the performance tracking veriables
                    if plotTraining:
                        totalReward = 0
                    # Interact with the training environment until termination
                    while done == 0:
                        # Choose an action according to the RL policy and the current RL state
                        action = self.chooseAction(state)
                        # Interact with the environment with the chosen action
                        nextState, reward, done, info = trainingEnvList[i].step(action)
                        # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                        reward = self.processReward(reward)
                        nextState = self.processState(nextState, coefficients)
                        self.buffer.rewards.append(reward)
                        self.buffer.is_terminals.append(done)
                        # Execute the learning procedure
                        stepsCounter += 1
                        if stepsCounter == batch_size:
                            self.learningBatch(batch_size)
                            stepsCounter = 0
                        # Update the RL state
                        state = nextState
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
            self.policy.eval()

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
                self.plotTraining(score[i], marketSymbol, displayOption=plotTraining)
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
        interactiveDisplayManager = DisplayManager(displayOptions=DisplayOption(False, False, True, rendering.recordVideo), 
                                                   figsize=default_fig_size) if interactiveTradingGraph or rendering.recordVideo else None
        # Interact with the environment until the episode termination
        while done == 0:
            # Choose an action according to the RL policy and the current RL state
            action = self.chooseAction(state, append=False)
            # Interact with the environment with the chosen action
            nextState, _, done, _ = testingEnvSmoothed.step(action)
            testingEnv.step(action)
            # Update the new state
            state = self.processState(nextState, coefficients)
            if interactiveDisplayManager:
                testingEnv.render(_displayManager=interactiveDisplayManager)
        # If required, show the rendering of the trading environment
        if rendering:
            testingEnv.render(displayOptions=rendering)
        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance(f'{self.strategyName} (Testing)')
        return testingEnv
    
    def saveModel(self, fileName):
        """
        GOAL: Save the RL policy, which is the policy Deep Neural Network.
        
        INPUTS: - fileName: Name of the file.
        
        OUTPUTS: /
        """
        torch.save(self.policy_old.state_dict(), fileName)


    def loadModel(self, fileName):
        """
        GOAL: Load a RL policy, which is the policy Deep Neural Network.
        
        INPUTS: - fileName: Name of the file.
        
        OUTPUTS: /
        """
        self.policy_old.load_state_dict(torch.load(fileName, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(fileName, map_location=lambda storage, loc: storage))