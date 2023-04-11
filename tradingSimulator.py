# coding=utf-8

"""
Goal: Implement a trading simulator to simulate and compare trading strategies.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
Modified 08/2021 by Alessandro Pavesi
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import os
import sys
import importlib
import pickle
import itertools
import yaml

import numpy as np
import pandas as pd

from tabulate import tabulate
from tqdm import tqdm
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from Environment.tradingEnv import TradingEnv
from DataProcessing.tradingPerformance import PerformanceEstimator
from DataProcessing.timeSeriesAnalyser import TimeSeriesAnalyser
from DRL.TDQN import TDQN
from Misc.displayManager import *

###############################################################################
########################### Class TradingSimulator ############################
###############################################################################

class TradingSimulator:
    """
    GOAL: Accurately simulating multiple trading strategies on different stocks
          to analyze and compare their performance.
        
    VARIABLES: /
          
    METHODS:   - displayTestbench: Display consecutively all the stocks
                                   included in the testbench.
               - analyseTimeSeries: Perform a detailled analysis of the stock
                                    market price time series.
               - plotEntireTrading: Plot the entire trading activity, with both
                                    the training and testing phases rendered on
                                    the same graph.
               - simulateNewStrategy: Simulate a new trading strategy on a 
                                      a certain stock of the testbench.
               - simulateExistingStrategy: Simulate an already existing
                                           trading strategy on a certain
                                           stock of the testbench.
               - evaluateStrategy: Evaluate a trading strategy on the
                                   entire testbench.
               - evaluateStock: Compare different trading strategies
                                on a certain stock of the testbench.
    """
    def __init__(self, configsFile='./Configurations/env-config.yml'):
        # 0. SET VARIABLES FROM CONFIG
        with open(configsFile, 'r') as yamlfile:
            self.configs = yaml.safe_load(yamlfile)
        environment_params = self.configs["environment"]
        self.startingDate = environment_params["startingDate"]
        self.endingDate = environment_params["endingDate"]
        self.splittingDate = environment_params["splittingDate"]
        self.actionSpace = environment_params["actionSpace"]
        self.money = environment_params["money"]
        self.stateLength = environment_params["stateLength"]
        self.bounds = environment_params["bounds"]
        self.step = environment_params["step"]
        self.numberOfEpisodes = environment_params["numberOfEpisodes"]
        self.verbose = environment_params["verbose"]
        self.plotTraining = environment_params["plotTraining"]
        self.rendering = environment_params["rendering"]
        self.showPerformance = environment_params["showPerformance"]
        self.saveStrategy = environment_params["saveStrategy"]
        self.fictives = environment_params["fictives"]
        self.strategies = environment_params["strategies"]
        self.stocks = environment_params["stocks"]
        self.indices = environment_params["indices"]
        self.companies = environment_params["companies"]
        self.strategies = environment_params["strategies"]
        self.strategiesAI = environment_params["strategiesAI"]
        self.percentageCosts = environment_params["percentageCosts"]
        self.context = environment_params["context"]
        # Variables setting up the default transaction costs
        self.transactionCosts = self.percentageCosts[1] / 100


    def getTradingStrategy(self, strategyName, **extraModelArgs):
        # Retrieve the trading strategy information
        if(strategyName in self.strategies):
            strategy = self.strategies[strategyName]
            trainingParameters = [self.bounds, self.step]
            ai = False
        elif(strategyName in self.strategiesAI):
            strategy = self.strategiesAI[strategyName]
            trainingParameters = [self.numberOfEpisodes]
            ai = True
        # Error message if the strategy specified is not valid or not supported
        else:
            print("The strategy specified is not valid, only the following strategies are supported:")
            for strategy in self.strategies:
                print("".join(['- ', strategy]))
            for strategy in self.strategiesAI:
                print("".join(['- ', strategy]))
            raise SystemError("Please check the trading strategy specified.")
        # Instanciate the strategy classes
        if ai:
            strategyModule = importlib.import_module(f'DRL.{str(strategy)}')
            className = getattr(strategyModule, strategy)
            tradingStrategy = className(self.observationSpace, self.actionSpace, **extraModelArgs)
        else:
            strategyModule = importlib.import_module('Misc.classicalStrategy')
            className = getattr(strategyModule, strategy)
            tradingStrategy = className(**extraModelArgs)
        return tradingStrategy, trainingParameters
    

    def getStock(self, stockName):
        # Retrieve the trading stock information
        stock = stockName
        if(stockName in self.fictives):
            stock = self.fictives[stockName]
        elif(stockName in self.indices):
            stock = self.indices[stockName]
        elif(stockName in self.companies):
            stock = self.companies[stockName]
        return stock
    

    def simulateNewStrategy(self, 
                            strategyName, 
                            stockName,
                            money=10000,
                            batch_mode=False, 
                            batch_size=64, 
                            interactiveTrain=False, 
                            trainShowPerformance=False, 
                            trainPlot=False, 
                            plotTrainEnv=False, 
                            interactiveTest=False, 
                            testShowPerformance=False, 
                            testOnLiveData=False, 
                            testPlotQValues=False,
                            trainTestRendering=False,
                            saveStrategy=True,
                            **extraModelArgs):
        # 1. INIT PHASE
        context = {} if strategyName == "PPO" or strategyName == "TDCQN" else self.context
        self.observationSpace = 1 + (self.stateLength - 1) * (4 + len(context))
        stock = self.getStock(stockName)
        tradingStrategy, trainingParameters = self.getTradingStrategy(strategyName, **extraModelArgs)
        # 2. TRAINING PHASE
        # Initialize the trading environment associated with the training phase
        print("=================================== TRAINING PHASE ===================================")
        trainingEnv = TradingEnv(stock, self.startingDate, self.splittingDate, money, context, self.stateLength, self.transactionCosts)
        # Training of the trading strategy
        if batch_mode:
            trainingEnv = tradingStrategy.trainingBatch(trainingEnv, 
                                                        context,
                                                        trainingParameters=trainingParameters,
                                                        batch_size=batch_size,
                                                        verbose=True, 
                                                        rendering=DisplayOption(False, plotTrainEnv, False),
                                                        plotTraining=DisplayOption(False, trainPlot, False), 
                                                        showPerformance=trainShowPerformance,
                                                        interactiveTradingGraph=interactiveTrain)
        else:
            trainingEnv = tradingStrategy.training(trainingEnv, 
                                                   context,
                                                   trainingParameters=trainingParameters,
                                                   verbose=True, 
                                                   rendering=DisplayOption(False, plotTrainEnv, False),
                                                   plotTraining=DisplayOption(False, trainPlot, False), 
                                                   showPerformance=trainShowPerformance,
                                                   interactiveTradingGraph=interactiveTrain)
        # 3. TESTING PHASE
        # Initialize the trading environment associated with the testing phase
        print("=================================== TESTING PHASE ===================================")
        testingEnv = TradingEnv(stock, self.splittingDate, self.endingDate, money, context, self.stateLength, self.transactionCosts, liveData=testOnLiveData)
        # Testing of the trading strategy
        testingEnv = tradingStrategy.testing(trainingEnv,
                                             testingEnv,
                                             rendering=DisplayOption(False, testPlotQValues, False),
                                             showPerformance=testShowPerformance,
                                             interactiveTradingGraph=interactiveTest)
        # Show the entire unified rendering of the training and testing phases
        if trainTestRendering:
            self.plotEntireTrading(trainingEnv, testingEnv)
        # 4. TERMINATION PHASE
        # If required, save the trading strategy with Pickle
        if (saveStrategy):
            fileName = f"Strategies/{tradingStrategy.strategyName}_{stock}_{self.startingDate}_{self.splittingDate}.model"
            tradingStrategy.saveModel(fileName)
        # Return of the trading strategy simulated and of the trading environments backtested
        return tradingStrategy, trainingEnv, testingEnv


    def simulateMultipleStrategy(self, 
                                 strategyName, 
                                 stockNames,
                                 money=10000,
                                 batch_mode=False, 
                                 batch_size=64, 
                                 interactiveTrain=False, 
                                 trainShowPerformance=False, 
                                 trainPlot=False, 
                                 plotTrainEnv=False, 
                                 interactiveTest=False, 
                                 testShowPerformance=False, 
                                 testOnLiveData=False, 
                                 testPlotQValues=False,
                                 trainTestRendering=False,
                                 saveStrategy=True,
                                 **extraModelArgs):
        """
        GOAL: Simulate a new trading strategy on a list of stocks included in the
              testbench, with both learning and testing phases.
        """
        # 1. INIT PHASE
        context = {} if strategyName == "PPO" or strategyName == "TDCQN" else self.context
        self.observationSpace = 1 + (self.stateLength - 1) * (4+len(context))
        tradingStrategy, trainingParameters = self.getTradingStrategy(strategyName, **extraModelArgs)
        tradingStrategies, trainingEnvs, testingEnvs = [], [], []

        for stockName in stockNames:
            stock = self.getStock(stockName)
            # 2. TRAINING PHASE
            # Initialize the trading environment associated with the training phase
            print("=================================== TRAINING PHASE ===================================")
            trainingEnv = TradingEnv(stock, self.startingDate, self.splittingDate, money, context, self.stateLength, self.transactionCosts)
            # Training of the trading strategy
            if batch_mode:
                trainingEnv = tradingStrategy.trainingBatch(trainingEnv, 
                                                            context,
                                                            trainingParameters=trainingParameters,
                                                            batch_size=batch_size,
                                                            verbose=True, 
                                                            rendering=DisplayOption(False, plotTrainEnv, False),
                                                            plotTraining=DisplayOption(False, trainPlot, False), 
                                                            showPerformance=trainShowPerformance,
                                                            interactiveTradingGraph=interactiveTrain)
            else:
                trainingEnv = tradingStrategy.training(trainingEnv, 
                                                       context,
                                                       trainingParameters=trainingParameters,
                                                       verbose=True, 
                                                       rendering=DisplayOption(False, plotTrainEnv, False),
                                                       plotTraining=DisplayOption(False, trainPlot, False), 
                                                       showPerformance=trainShowPerformance,
                                                       interactiveTradingGraph=interactiveTrain)
            # 3. TESTING PHASE
            # Initialize the trading environment associated with the testing phase
            print("=================================== TESTING PHASE ===================================")
            testingEnv = TradingEnv(stock, self.splittingDate, self.endingDate, money, context, self.stateLength, self.transactionCosts, liveData=testOnLiveData)
            # Testing of the trading strategy
            testingEnv = tradingStrategy.testing(trainingEnv,
                                                testingEnv,
                                                rendering=DisplayOption(False, testPlotQValues, False),
                                                showPerformance=testShowPerformance,
                                                interactiveTradingGraph=interactiveTest)
            # Show the entire unified rendering of the training and testing phases
            if trainTestRendering:
                self.plotEntireTrading(trainingEnv, testingEnv)
            # 4. TERMINATION PHASE
            # If required, save the trading strategy with Pickle
            if (saveStrategy):
                fileName = f"Strategies/{tradingStrategy.strategyName}_{stock}_{self.startingDate}_{self.splittingDate}.model"
                tradingStrategy.saveModel(fileName)
            tradingStrategies.append(tradingStrategy)
            trainingEnvs.append(trainingEnv)
            testingEnvs.append(testingEnv)

        # Return of the trading strategy simulated and of the trading environments backtested
        return tradingStrategies, trainingEnvs, testingEnvs


    def simulateExistingStrategy(self, 
                                 strategyName, 
                                 stockName,
                                 money=10000,
                                 interactiveTest=False, 
                                 testShowPerformance=False, 
                                 testOnLiveData=False, 
                                 testPlotQValues=False,
                                 trainTestRendering=False,
                                 **extraModelArgs):
        """
        GOAL: Simulate an already existing trading strategy on a certain
              stock of the testbench, the strategy being loaded from the
              strategy dataset. There is no training phase, only a testing
              phase.
        
        INPUTS: - strategyName: Name of the trading strategy.
                - stockName: Name of the stock (in the testbench).
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splittingDate: Spliting date between the training dataset
                                and the testing dataset.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - tradingStrategy: Trading strategy simulated.
                 - trainingEnv: Trading environment related to the training phase.
                 - testingEnv: Trading environment related to the testing phase.
        """
        # 1. INIT PHASE
        context = {} if strategyName == "PPO" or strategyName == "TDCQN" else self.context
        stock = self.getStock(stockName)
        tradingStrategy, trainingParameters = self.getTradingStrategy(strategyName, **extraModelArgs)
        fileName = f"Strategies/{tradingStrategy.strategyName}_{stock}_{self.startingDate}_{self.splittingDate}.model"
        exists = os.path.isfile(fileName)
        # If affirmative, load the trading strategy
        if exists:
            tradingStrategy.loadModel(fileName)
        else:
            raise SystemError("The trading strategy specified does not exist, please provide a valid one.")
    
        # 3. TESTING PHASE
        # Initialize the trading environments associated with the testing phase
        trainingEnv = TradingEnv(stock, self.startingDate, self.splittingDate, money, context, self.stateLength, self.transactionCosts)
        testingEnv = TradingEnv(stock, self.splittingDate, self.endingDate, money, context, self.stateLength, self.transactionCosts, liveData=testOnLiveData)
        # Testing of the trading strategy
        testingEnv = tradingStrategy.testing(trainingEnv,
                                            testingEnv,
                                            rendering=DisplayOption(False, testPlotQValues, False),
                                            showPerformance=testShowPerformance,
                                            interactiveTradingGraph=interactiveTest)

        # Show the entire unified rendering of the training and testing phases
        if trainTestRendering:
            self.plotEntireTrading(trainingEnv, testingEnv)
        return tradingStrategy, trainingEnv, testingEnv

    def evaluateStrategy(self, 
                         strategyName, 
                         stockName,
                         money=10000,
                         batch_mode=False, 
                         batch_size=64, 
                         interactiveTrain=False, 
                         trainShowPerformance=False, 
                         trainPlot=False, 
                         plotTrainEnv=False, 
                         interactiveTest=False, 
                         testShowPerformance=False, 
                         testOnLiveData=False, 
                         testPlotQValues=False,
                         trainTestRendering=False,
                         saveStrategy=True):
        """
        GOAL: Evaluate the performance of a trading strategy on the entire
              testbench of stocks designed.
        OUTPUTS: - performanceTable: Table summarizing the performance of
                                     a trading strategy.
        """
        # Initialization of some variables
        performanceTable = [["Profit & Loss (P&L)"], ["Annualized Return"], ["Annualized Volatility"], ["Sharpe Ratio"],
                            ["Sortino Ratio"], ["Maximum DrawDown"], ["Maximum DrawDown Duration"], ["Profitability"],
                            ["Ratio Average Profit/Loss"], ["Skewness"]]
        headers = ["Performance Indicator"]
        # Loop through each stock included in the testbench (progress bar)
        print("Trading strategy evaluation progression:")
        # for stock in tqdm(itertools.chain(indices, companies)):
        for stock in tqdm(self.stocks):
            # Simulation of the trading strategy on the current stock
            try:
                # Simulate an already existing trading strategy on the current stock
                _, _, testingEnv = self.simulateExistingStrategy(strategyName, 
                                                                 stockName, 
                                                                 money, 
                                                                 interactiveTest,  
                                                                 testShowPerformance,  
                                                                 testOnLiveData,  
                                                                 testPlotQValues, 
                                                                 trainTestRendering)
            except SystemError:
                # Simulate a new trading strategy on the current stock
                _, _, testingEnv = self.simulateNewStrategy(strategyName, 
                                                            stockName,
                                                            money,
                                                            batch_mode, 
                                                            batch_size, 
                                                            interactiveTrain, 
                                                            trainShowPerformance, 
                                                            trainPlot, 
                                                            plotTrainEnv, 
                                                            interactiveTest, 
                                                            testShowPerformance, 
                                                            testOnLiveData, 
                                                            testPlotQValues,
                                                            trainTestRendering,
                                                            saveStrategy)
            # Retrieve the trading performance associated with the trading strategy
            analyser = PerformanceEstimator(testingEnv.data)
            performance = analyser.computePerformance()
            # Get the required format for the display of the performance table
            headers.append(stock)
            for i in range(len(performanceTable)):
                performanceTable[i].append(performance[i][1])
        # Display the performance table computed
        tabulation = tabulate(performanceTable, headers, tablefmt="fancy_grid", stralign="center")
        print(tabulation)
        # Computation of the average Sharpe Ratio (default performance indicator)
        sharpeRatio = np.mean([float(item) for item in performanceTable[3][1:]])
        print("Average Sharpe Ratio: " + "{0:.3f}".format(sharpeRatio))
        return performanceTable


    def evaluateStock(self, 
                      strategyName, 
                      stockName,
                      money=10000,
                      batch_mode=False, 
                      batch_size=64, 
                      interactiveTrain=False, 
                      trainShowPerformance=False, 
                      trainPlot=False, 
                      plotTrainEnv=False, 
                      interactiveTest=False, 
                      testShowPerformance=False, 
                      testOnLiveData=False, 
                      testPlotQValues=False,
                      trainTestRendering=False,
                      saveStrategy=True):

        """
        GOAL: Simulate and compare the performance achieved by all the supported
              trading strategies on a certain stock of the testbench.
        
        INPUTS: - stockName: Name of the stock (in the testbench).
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splittingDate: Spliting date between the training dataset
                                and the testing dataset.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - bounds: Bounds of the parameter search space (training).
                - step: Step of the parameter search space (training).
                - numberOfEpisodes: Number of epsiodes of the RL training phase.
                - verbose: Enable the printing of a simulation feedback.
                - plotTraining: Enable the plotting of the training results.
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
                - saveStrategy: Enable the saving of the trading strategy.
        
        OUTPUTS: - performanceTable: Table summarizing the performance of
                                     a trading strategy.
        """

        # Initialization of some variables
        performanceTable = [["Profit & Loss (P&L)"], ["Annualized Return"], ["Annualized Volatility"], ["Sharpe Ratio"],
                            ["Sortino Ratio"], ["Maximum DrawDown"], ["Maximum DrawDown Duration"], ["Profitability"],
                            ["Ratio Average Profit/Loss"], ["Skewness"]]
        headers = ["Performance Indicator"]
        # Loop through all the trading strategies supported (progress bar)
        print("Trading strategies evaluation progression:")
        for strategy in tqdm(itertools.chain(self.strategies, self.strategiesAI)):
            # Simulation of the trading strategy on the current stock
            try:
                # Simulate an already existing trading strategy on the current stock
                _, _, testingEnv = self.simulateExistingStrategy(strategyName, 
                                                                 stockName, 
                                                                 money, 
                                                                 interactiveTest,  
                                                                 testShowPerformance,  
                                                                 testOnLiveData,  
                                                                 testPlotQValues, 
                                                                 trainTestRendering)
            except SystemError:
                # Simulate a new trading strategy on the current stock
                _, _, testingEnv = self.simulateNewStrategy(strategyName, 
                                                            stockName,
                                                            money,
                                                            batch_mode, 
                                                            batch_size, 
                                                            interactiveTrain, 
                                                            trainShowPerformance, 
                                                            trainPlot, 
                                                            plotTrainEnv, 
                                                            interactiveTest, 
                                                            testShowPerformance, 
                                                            testOnLiveData, 
                                                            testPlotQValues,
                                                            trainTestRendering,
                                                            saveStrategy)
            # Retrieve the trading performance associated with the trading strategy
            analyser = PerformanceEstimator(testingEnv.data)
            performance = analyser.computePerformance()
            # Get the required format for the display of the performance table
            headers.append(strategy)
            for i in range(len(performanceTable)):
                performanceTable[i].append(performance[i][1])
        # Display the performance table
        tabulation = tabulate(performanceTable, headers, tablefmt="fancy_grid", stralign="center")
        print(tabulation)
        return performanceTable

    def displayTestbench(self):
        """
        GOAL: Display consecutively all the stocks included in the
              testbench (trading indices and companies).
        
        INPUTS: - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
        
        OUTPUTS: /
        """
        # Display the stocks included in the testbench (trading indices)
        for _, stock in self.indices.items():
            env = TradingEnv(stock, self.startingDate, self.endingDate, 0)
            env.render()
        # Display the stocks included in the testbench (companies)
        for _, stock in self.companies.items():
            env = TradingEnv(stock, self.startingDate, self.endingDate, 0)
            env.render()


    def analyseTimeSeries(self, stockName):
        """
        GOAL: Perform a detailed analysis of the stock market
              price time series.
        
        INPUTS: - stockName: Name of the stock (in the testbench).
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splittingDate: Splitting date between the training dataset
                                and the testing dataset.
        
        OUTPUTS: /
        """
        stock = self.getStock(stockName)
        # TRAINING DATA
        print("\n\n\nAnalysis of the TRAINING phase time series")
        print("------------------------------------------\n")
        trainingEnv = TradingEnv(stock, self.startingDate, self.splittingDate, 0)
        timeSeries = trainingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()
        # TESTING DATA
        print("\n\n\nAnalysis of the TESTING phase time series")
        print("------------------------------------------\n")
        testingEnv = TradingEnv(stock, self.splittingDate, self.endingDate, 0)
        timeSeries = testingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()
        # ENTIRE TRADING DATA
        print("\n\n\nAnalysis of the entire time series (both training and testing phases)")
        print("---------------------------------------------------------------------\n")
        tradingEnv = TradingEnv(stock, self.startingDate, self.endingDate, 0)
        timeSeries = tradingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()


    def plotEntireTrading(self, trainingEnv, testingEnv, strategyName):
        """
        GOAL: Plot the entire trading activity, with both the training
              and testing phases rendered on the same graph for
              comparison purposes.
        
        INPUTS: - trainingEnv: Trading environment for training.
                - testingEnv: Trading environment for testing.
        
        OUTPUTS: /
        """
        # Artificial trick to assert the continuity of the Money curve
        ratio = trainingEnv.data['Money'][-1] / testingEnv.data['Money'][0]
        testingEnv.data['Money'] = ratio * testingEnv.data['Money']
        # Concatenation of the training and testing trading dataframes
        dataframes = [trainingEnv.data, testingEnv.data]
        data = pd.concat(dataframes)
        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)
        # Plot the first graph -> Evolution of the stock market price
        trainingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2)
        testingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2, label='_nolegend_')
        ax1.plot(data.loc[data['Action'] == 1.0].index,
                 data['Close'][data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax1.plot(data.loc[data['Action'] == -1.0].index,
                 data['Close'][data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        # Plot the second graph -> Evolution of the trading capital
        trainingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2)
        testingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2, label='_nolegend_')
        ax2.plot(data.loc[data['Action'] == 1.0].index,
                 data['Money'][data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax2.plot(data.loc[data['Action'] == -1.0].index,
                 data['Money'][data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        # Plot the vertical line seperating the training and testing datasets
        ax1.axvline(pd.Timestamp(self.splittingDate), color='black', linewidth=2.0)
        ax2.axvline(pd.Timestamp(self.splittingDate), color='black', linewidth=2.0)
        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long", "Short", "Train/Test separation"])
        ax2.legend(["Capital", "Long", "Short", "Train/Test separation"])
        plt.savefig(f'Figures/{str(trainingEnv.marketSymbol)}_{strategyName}_TrainingTestingRendering.png')
        # plt.show()