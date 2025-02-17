# coding=utf-8

"""
Goal: Implement a trading environment compatible with OpenAI Gym.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import os
import gym
import math
import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None

from matplotlib import pyplot as plt

from DataProcessing.dataDownloader import AlphaVantage
from DataProcessing.dataDownloader import YahooFinance
from DataProcessing.dataDownloader import YahooFin
from DataProcessing.dataDownloader import CSVHandler
from DataProcessing.fictiveStockGenerator import StockGenerator

from Misc.displayManager import *
from yahoo_fin import stock_info
from datetime import datetime
import time


###############################################################################
################################ Global variables #############################
###############################################################################

# Boolean handling the saving of the stock market data downloaded
saving = True

# Variable related to the fictive stocks supported
fictiveStocks = ('LINEARUP', 'LINEARDOWN', 'SINUSOIDAL', 'TRIANGLE')


###############################################################################
############################## Class TradingEnv ###############################
###############################################################################

class TradingEnv(gym.Env):
    """
    GOAL: Implement a custom trading environment compatible with OpenAI Gym.
    
    VARIABLES:  - data: Dataframe monitoring the trading activity.
                - state: RL state to be returned to the RL agent.
                - reward: RL reward to be returned to the RL agent.
                - done: RL episode termination signal.
                - t: Current trading time step.
                - marketSymbol: Stock market symbol.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - stateLength: Number of trading time steps included in the state.
                - numberOfShares: Number of shares currently owned by the agent.
                - transactionCosts: Transaction costs associated with the trading
                                    activity (e.g. 0.01 is 1% of loss).
                                
    METHODS:    - __init__: Object constructor initializing the trading environment.
                - reset: Perform a soft reset of the trading environment.
                - step: Transition to the next trading time step.
                - render: Illustrate graphically the trading environment.
    """
    def __init__(self, 
                 marketSymbol, 
                 startingDate, 
                 endingDate, 
                 money, 
                 context={}, 
                 stateLength=30,
                 transactionCosts=0, 
                 startingPoint=0, 
                 liveData=False):
        """
        GOAL: Object constructor initializing the trading environment by setting up
              the trading activity dataframe as well as other important variables.
        
        INPUTS: - marketSymbol: Stock market symbol.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - money: Initial amount of money at the disposal of the agent.
                - stateLength: Number of trading time steps included in the RL state.
                - transactionCosts: Transaction costs associated with the trading
                                    activity (e.g. 0.01 is 1% of loss).
                - startingPoint: Optional starting point (iteration) of the trading activity.
        
        OUTPUTS: /
        """
        # CASE 1: Fictive stock generation
        if (marketSymbol in fictiveStocks):
            stockGeneration = StockGenerator()
            if (marketSymbol == 'LINEARUP'):
                self.data = stockGeneration.linearUp(startingDate, endingDate)
            elif (marketSymbol == 'LINEARDOWN'):
                self.data = stockGeneration.linearDown(startingDate, endingDate)
            elif (marketSymbol == 'SINUSOIDAL'):
                self.data = stockGeneration.sinusoidal(startingDate, endingDate)
            else:
                self.data = stockGeneration.triangle(startingDate, endingDate)

        # CASE 2: Real stock loading
        else:
            # Check if the stock market data is already present in the database
            csvConverter = CSVHandler()
            csvName = "".join(['Data/', marketSymbol, '_', startingDate, '_', endingDate])
            exists = os.path.isfile(csvName + '.csv')

            # If affirmative, load the stock market data from the database
            if (exists):
                self.data = csvConverter.CSVToDataframe(csvName)
            # Otherwise, download the stock market data from Yahoo Finance and save it in the database
            else:
                downloader1 = YahooFinance()
                downloader2 = AlphaVantage()
                try:
                    self.data = downloader1.getDailyData(marketSymbol, startingDate, endingDate)
                except:
                    self.data = downloader2.getDailyData(marketSymbol, startingDate, endingDate)
                if saving == True: csvConverter.dataframeToCSV(csvName, self.data)
        
        # Set the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.
        # Set the RL variables common to every OpenAI gym environments
        self.reward = 0.
        self.done = 0
        # Set additional variables related to the trading activity
        self.marketSymbol = marketSymbol
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.stateLength = stateLength
        self.t = stateLength
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts
        self.epsilon = 0.1
        self.liveData = liveData
        self.timestamp = None
        # If required, set a custom starting point for the trading activity
        if startingPoint: self.setStartingPoint(startingPoint)

        ### add context
        context_symbols = context.values()
        for symbol in context_symbols:
            if not os.path.exists('./Context/'):
                os.makedirs('./Context/')
            csvName = "".join(['./Context/', symbol, '_', startingDate, '_', endingDate])
            exists = os.path.isfile(csvName + '.csv')
            # If affirmative, load the stock market data from the database
            if(exists):
                context_series = csvConverter.CSVToDataframe(csvName)
            # Otherwise, download the stock market data from Yahoo Finance and save it in the database
            else:  
                downloader = YahooFin()
                context_series = downloader.getDailyData(symbol, startingDate, endingDate)
                if saving == True: csvConverter.dataframeToCSV(csvName, context_series)
            # Pre-process data
            context_series = context_series.reindex(self.data.index)
            context_series = context_series['Close'].to_frame()
            context_series = context_series.add_suffix(f'_{symbol}')
            self.data = pd.concat([self.data, context_series], axis=1)
        
        # list of lists
        base_state = self.data[['Close','Low','High','Volume']].iloc[0:stateLength].T.values.tolist()
        context_state = self.data.filter(regex='^Close_', axis=1).iloc[0:stateLength].T.values.tolist()
        self.state = base_state+context_state+[[0]]
        

    def reset(self):
        """
        GOAL: Perform a soft reset of the trading environment. 
        
        INPUTS: /    
        
        OUTPUTS: - state: RL state returned to the trading strategy.
        """
        # Reset the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.
        # Reset the RL variables common to every OpenAI gym environments
        base_state = self.data[['Close','Low','High','Volume']].iloc[0:self.stateLength].T.values.tolist()
        context_state = self.data.filter(regex='^Close_',axis=1).iloc[0:self.stateLength].T.values.tolist()
        self.state = base_state+context_state+[[0]]
        self.reward = 0.
        self.done = 0
        # Reset additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0
        return self.state


    def computeLowerBound(self, cash, numberOfShares, price):
        """
        GOAL: Compute the lower bound of the complete RL action space, 
              i.e. the minimum number of share to trade.
        
        INPUTS: - cash: Value of the cash owned by the agent.
                - numberOfShares: Number of shares owned by the agent.
                - price: Last price observed.
        
        OUTPUTS: - lowerBound: Lower bound of the RL action space.
        """
        # Computation of the RL action lower bound
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound


    def step(self, action):
        """
        GOAL: Transition to the next trading time step based on the
              trading position decision made (either long or short).
        
        INPUTS: - action: Trading decision (1 = long, 0 = short).    
        
        OUTPUTS: - state: RL state to be returned to the RL agent.
                 - reward: RL reward to be returned to the RL agent.
                 - done: RL episode termination signal (boolean).
                 - info: Additional information returned to the RL agent.
        """

        # Setting of some local variables
        t = self.t
        numberOfShares = self.numberOfShares
        customReward = False

        # CASE 1: LONG POSITION
        if (action == 1):
            self.data['Position'][t] = 1
            # Case a: Long -> Long
            if (self.data['Position'][t - 1] == 1):
                self.data['Cash'][t] = self.data['Cash'][t - 1]
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
            # Case b: No position -> Long
            elif (self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(
                    self.data['Cash'][t - 1] / (self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (
                            1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = 1
            # Case c: Short -> Long
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (
                            1 + self.transactionCosts)
                self.numberOfShares = math.floor(
                    self.data['Cash'][t] / (self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] - self.numberOfShares * self.data['Close'][t] * (
                            1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = 1

        # CASE 2: SHORT POSITION
        elif (action == 0):
            self.data['Position'][t] = -1
            # Case a: Short -> Short
            if (self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares,
                                                    self.data['Close'][t - 1])
                if lowerBound <= 0:
                    self.data['Cash'][t] = self.data['Cash'][t - 1]
                    self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), self.numberOfShares)
                    self.numberOfShares -= numberOfSharesToBuy
                    self.data['Cash'][t] = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (
                                1 + self.transactionCosts)
                    self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                    customReward = True
            # Case b: No position -> Short
            elif (self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(
                    self.data['Cash'][t - 1] / (self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (
                            1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1
            # Case c: Long -> Short
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (
                            1 - self.transactionCosts)
                self.numberOfShares = math.floor(
                    self.data['Cash'][t] / (self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] + self.numberOfShares * self.data['Close'][t] * (
                            1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1

        # CASE 3: PROHIBITED ACTION
        else:
            raise SystemExit("Prohibited action! Action should be either 1 (long) or 0 (short).")

        # Update the total amount of money owned by the agent, as well as the return generated
        self.data['Money'][t] = self.data['Holdings'][t] + self.data['Cash'][t]
        self.data['Returns'][t] = (self.data['Money'][t] - self.data['Money'][t - 1]) / self.data['Money'][t - 1]

        # Set the RL reward returned to the trading agent
        if not customReward:
            self.reward = self.data['Returns'][t]
        else:
            self.reward = (self.data['Close'][t - 1] - self.data['Close'][t]) / self.data['Close'][t - 1]

        # Transition to the next trading time step
        self.t = self.t + 1
        # Set the RL variables common to every OpenAI gym environments
        columns = ['Close','Low','High','Volume']+[i for i in self.data.columns if 'Close_' in i]
        self.state = self.data[columns].iloc[self.t - self.stateLength : self.t].T.values.tolist() + [ [self.data['Position'][self.t - 1]] ]
        
        if (self.t == self.data.shape[0]):
            if (self.liveData):
                timestamp, stock_data  = self.getLiveData(self.data.iloc[t].to_dict())
                if (self.timestamp and self.timestamp == timestamp and not self.done):
                    # print("Market is closed !")
                    self.done = 1
                else:
                    self.data.loc[timestamp] = stock_data
                    self.timestamp = timestamp
                    display(self.data)
                    time.sleep(15)
            else:
                self.done = 1

        # Same reasoning with the other action (exploration trick)
        otherAction = int(not bool(action))
        customReward = False
        if (otherAction == 1):
            otherPosition = 1
            if (self.data['Position'][t - 1] == 1):
                otherCash = self.data['Cash'][t - 1]
                otherHoldings = numberOfShares * self.data['Close'][t]
            elif (self.data['Position'][t - 1] == 0):
                numberOfShares = math.floor(
                    self.data['Cash'][t - 1] / (self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] - numberOfShares * self.data['Close'][t] * (
                            1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'][t]
            else:
                otherCash = self.data['Cash'][t - 1] - numberOfShares * self.data['Close'][t] * (
                            1 + self.transactionCosts)
                numberOfShares = math.floor(otherCash / (self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = otherCash - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'][t]
        else:
            otherPosition = -1
            if (self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares,
                                                    self.data['Close'][t - 1])
                if lowerBound <= 0:
                    otherCash = self.data['Cash'][t - 1]
                    otherHoldings = - numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), numberOfShares)
                    numberOfShares -= numberOfSharesToBuy
                    otherCash = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (
                                1 + self.transactionCosts)
                    otherHoldings = - numberOfShares * self.data['Close'][t]
                    customReward = True
            elif (self.data['Position'][t - 1] == 0):
                numberOfShares = math.floor(
                    self.data['Cash'][t - 1] / (self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] + numberOfShares * self.data['Close'][t] * (
                            1 - self.transactionCosts)
                otherHoldings = - numberOfShares * self.data['Close'][t]
            else:
                otherCash = self.data['Cash'][t - 1] + numberOfShares * self.data['Close'][t] * (
                            1 - self.transactionCosts)
                numberOfShares = math.floor(otherCash / (self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = otherCash + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                otherHoldings = - self.numberOfShares * self.data['Close'][t]
        otherMoney = otherHoldings + otherCash
        if not customReward:
            otherReward = (otherMoney - self.data['Money'][t - 1]) / self.data['Money'][t - 1]
        else:
            otherReward = (self.data['Close'][t - 1] - self.data['Close'][t]) / self.data['Close'][t - 1]
        # Set the RL variables common to every OpenAI gym environments
        columns = ['Close','Low','High','Volume']+[i for i in self.data.columns if 'Close_' in i]
        otherState = self.data[columns].iloc[self.t - self.stateLength : self.t].T.values.tolist() + [[otherPosition]]
        self.info = {'State': otherState, 'Reward': otherReward, 'Done': self.done}
        # Return the trading environment feedback to the RL trading agent
        return self.state, self.reward, self.done, self.info


    def render(self, displayOptions=DisplayOption(), _displayManager=None, extraText=""):
        """
        GOAL: Illustrate graphically the trading activity, by plotting
              both the evolution of the stock market price and the 
              evolution of the trading capital. All the trading decisions
              (long and short positions) are displayed as well.
        
        INPUTS: /   
        
        OUTPUTS: /
        """
        # Set the Matplotlib figure and subplots
        displayManager = DisplayManager(displayOptions=displayOptions, figsize=(20.0, 10.0)) if not _displayManager else _displayManager
        ax1 = displayManager.add_subplot(311, ylabel='Price', xlabel='Time')
        ax2 = displayManager.add_subplot(312, ylabel='Capital', xlabel='Time', sharex=ax1)
        ax3 = displayManager.add_subplot(313, ylabel='Liquidity', xlabel='Time', sharex=ax1)

        timestamps = self.data.index[:self.t].to_numpy()
        # Plot the first graph -> Evolution of the stock market price
        displayManager.plot(ax1, 0, timestamps, self.data['Close'][:self.t].to_numpy(), color='blue', lw=1)
        displayManager.plot(ax1, 1, self.data.loc[self.data['Action'] == 1.0].index, 
                            self.data['Close'][self.data['Action'] == 1.0],
                            linestyle='None', marker='^', markersize=5, color='green')   
        displayManager.plot(ax1, 2, self.data.loc[self.data['Action'] == -1.0].index, 
                            self.data['Close'][self.data['Action'] == -1.0],
                            linestyle='None', marker='v', markersize=5, color='red')
        
        # Plot the second graph -> Evolution of the trading capital
        displayManager.plot(ax2, 0, timestamps, self.data['Money'][:self.t].to_numpy(), color='blue', lw=1)
        displayManager.plot(ax2, 1, self.data.loc[self.data['Action'] == 1.0].index, 
                            self.data['Money'][self.data['Action'] == 1.0],
                            linestyle='None', marker='^', markersize=5, color='green')   
        displayManager.plot(ax2, 2, self.data.loc[self.data['Action'] == -1.0].index, 
                            self.data['Money'][self.data['Action'] == -1.0],
                            linestyle='None', marker= 'v', markersize=5, color='red')
        
        # Plot the third graph -> Evolution of the liquid assets
        displayManager.plot(ax3, 0, timestamps, self.data['Cash'][:self.t].to_numpy(), color='blue', lw=1)
        displayManager.plot(ax3, 1, self.data.loc[self.data['Action'] == 1.0].index, 
                            self.data['Cash'][self.data['Action'] == 1.0],
                            linestyle='None', marker='^', markersize=5, color='green')   
        displayManager.plot(ax3, 2, self.data.loc[self.data['Action'] == -1.0].index, 
                            self.data['Cash'][self.data['Action'] == -1.0],
                            linestyle='None', marker='v', markersize=5, color='red')
        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short"])
        ax2.legend(["Capital", "Long", "Short"])
        ax3.legend(["Cash", "Long", "Short"])
        displayManager.show(f"{str(self.marketSymbol)}_{extraText}Rendering")


    def setStartingPoint(self, startingPoint):
        """
        GOAL: Setting an arbitrary starting point regarding the trading activity.
              This technique is used for better generalization of the RL agent.
        
        INPUTS: - startingPoint: Optional starting point (iteration) of the trading activity.
        
        OUTPUTS: /
        """
        # Setting a custom starting point
        self.t = np.clip(startingPoint, self.stateLength, len(self.data.index))
        columns = ['Close','Low','High','Volume']+[i for i in self.data.columns if 'Close_' in i]
        # Set the RL variables common to every OpenAI gym environments
        self.state = self.data[columns].iloc[self.t - self.stateLength : self.t].T.values.tolist() + [[self.data['Position'][self.t - 1]]]
        if self.t == self.data.shape[0]:
            self.done = 1


    def getLiveData(self, last={}):
        si = stock_info.get_quote_data(self.marketSymbol)
        stock_data = {
            **last,
            'Open': si['regularMarketOpen'], 
            'High': si['regularMarketDayHigh'], 
            'Low': si['regularMarketDayLow'], 
            'Close': si['regularMarketPrice'], 
            'Volume': si['regularMarketVolume'],
        }
        unix_epoch = si['regularMarketTime']
        timestamp = datetime.fromtimestamp(unix_epoch)
        return timestamp, stock_data 
