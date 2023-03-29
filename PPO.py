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
from torch.distributions import Categorical

import random
import copy
import datetime

import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from displayManager import *

from tradingPerformance import PerformanceEstimator
from dataAugmentation import DataAugmentation
from tradingEnv import TradingEnv


################################# PPO Policy ##################################
# Default parameters related to preprocessing
from tradingSimulator import endingDateCrypto

filterOrder = 5

# Default paramters related to the clipping of both the gradient and the RL rewards
gradientClipping = 1
rewardClipping = 1

learningUpdatePeriod = 1


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
    def __init__(self, observationSpace, actionSpace, numberOfNeurons):
        super(ActorCritic, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # actor
        self.actor = nn.Sequential(
            nn.Linear(observationSpace, numberOfNeurons),
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
            nn.Linear(numberOfNeurons, actionSpace),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(observationSpace, numberOfNeurons),
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
            nn.Linear(numberOfNeurons, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


###############################################################################
################################ Class PPO ###################################
###############################################################################

class PPO:
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

    def __init__(self, observationSpace, actionSpace, numberOfNeurons=512,
                 gamma=0.99, learningRateActor=0.0003, learningRateCritic=0.001,
                 K_epochs=80, eps_clip=0.2):

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

        # Initialise the random function with a new random seed
        random.seed(0)

        # Check availability of CUDA for the hardware (CPU or GPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Set the general parameters of the PPO algorithm
        self.learningRateActor = learningRateActor
        self.learningRateCritic = learningRateCritic

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(observationSpace, actionSpace, numberOfNeurons).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.learningRateActor},
            {'params': self.policy.critic.parameters(), 'lr': self.learningRateCritic}
        ])

        self.policy_old = ActorCritic(observationSpace, actionSpace, numberOfNeurons).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # Set both the observation and action spaces
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace

        # Initialization of the tensorboard writer
        self.writer = SummaryWriter('runs/' + datetime.datetime.now().strftime("%d/%m/%Y-%H.%M.%S"))

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
        closePrices = [state[0][i] for i in range(len(state[0]))]
        lowPrices = [state[1][i] for i in range(len(state[1]))]
        highPrices = [state[2][i] for i in range(len(state[2]))]
        volumes = [state[3][i] for i in range(len(state[3]))]

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

        # Process the state structure to obtain the appropriate format
        state = [item for sublist in state for item in sublist]

        return state

    def processReward(self, reward):
        """
        GOAL: Process the RL reward returned by the environment by clipping
              its value. Such technique has been shown to improve the stability
              the PPO algorithm.

        INPUTS: - reward: RL reward returned by the environment.

        OUTPUTS: - reward: Process RL reward.
        """

        return np.clip(reward, -rewardClipping, rewardClipping)

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

    def training(self, trainingEnv, 
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
            endingDate = '2020-1-1' if marketSymbol not in ['Binance', 'Litecoin', 'Cardano'] else endingDateCrypto
            money = trainingEnv.data['Money'][0]
            stateLength = trainingEnv.stateLength
            transactionCosts = trainingEnv.transactionCosts
            testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)
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

                        # # Trick for better exploration
                        # otherAction = int(not bool(action))
                        # otherReward = self.processReward(info['Reward'])
                        # otherNextState = self.processState(info['State'], coefficients)
                        # otherDone = info['Done']
                        # self.buffer.rewards.append(otherReward)
                        # self.buffer.is_terminals.append(otherDone)

                        # Execute the learning procedure
                        # stepsCounter += 1
                        # if stepsCounter == learningUpdatePeriod:
                        #     self.learning()
                        #     stepsCounter = 0

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
                self.plotTraining(score[i][:episode], marketSymbol, displayOption=plotTraining)

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('PPO (Training)')

        # Closing of the tensorboard writer
        self.writer.close()
        return trainingEnv

    def trainingBatch(self, 
                      trainingEnv, 
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
            testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)
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

                        # # Trick for better exploration
                        # otherAction = int(not bool(action))
                        # otherReward = self.processReward(info['Reward'])
                        # otherNextState = self.processState(info['State'], coefficients)
                        # otherDone = info['Done']
                        # self.buffer.rewards.append(otherReward)
                        # self.buffer.is_terminals.append(otherDone)

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
                self.plotTraining(score[i][:episode], marketSymbol, displayOption=plotTraining)

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('PPO (Training)')

        # Closing of the tensorboard writer
        self.writer.close()
        return trainingEnv

    def testing(self, trainingEnv, testingEnv, rendering=DisplayOption(), showPerformance=False, interactiveTradingGraph=False):
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
        testingEnvSmoothed = dataAugmentation.lowPassFilter(testingEnv, filterOrder)
        trainingEnv = dataAugmentation.lowPassFilter(trainingEnv, filterOrder)

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
            testingEnv.render()
            # self.plotQValues(QValues0, QValues1, testingEnv.marketSymbol)
        
        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance('PPO (Testing)')
        return testingEnv

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
        displayManager.show(f"{str(marketSymbol)}_Training_Results")

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
        displayManager.show(f"{str(marketSymbol)}_Training_Results")

    def plotQValues(self, QValues0, QValues1, marketSymbol, displayOption=DisplayOption(), extraTest=""):
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
        displayManager.show(f"{str(marketSymbol)}_{extraTest}_QValues")

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
                            if stepsCounter == learningUpdatePeriod:
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
                    self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=learningRate, weight_decay=L2Factor)
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
            displayManager.show(f"{str(marketSymbol)}_Training_Testing_Performance_{str(i+1)}")

        # Plot the expected performance of the intelligent DRL trading agent
        displayManager = DisplayManager(displayOptions=trainingTestingExpectedPerformanceDisplayOption)
        ax = displayManager.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
        ax.plot(expectedPerformanceTrain)
        ax.plot(expectedPerformanceTest)
        ax.fill_between(range(len(expectedPerformanceTrain)), expectedPerformanceTrain-stdPerformanceTrain, expectedPerformanceTrain+stdPerformanceTrain, alpha=0.25)
        ax.fill_between(range(len(expectedPerformanceTest)), expectedPerformanceTest-stdPerformanceTest, expectedPerformanceTest+stdPerformanceTest, alpha=0.25)
        ax.legend(["Training", "Testing"])
        displayManager.show(f"{str(marketSymbol)}_Training_Testing_Expected_Performance")

        # Closing of the tensorboard writer
        self.writer.close()
        return trainingEnv

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

