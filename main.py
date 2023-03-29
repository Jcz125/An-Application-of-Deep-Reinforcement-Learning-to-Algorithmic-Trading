# coding=utf-8

"""
Goal: Program Main.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
Modified 08/2021 by Alessandro Pavesi
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import argparse

from tradingSimulator import TradingSimulator, startingDate, splitingDate, endingDate, startingDateCrypto, splitingDateCrypto, endingDateCrypto

###############################################################################
##################################### MAIN ####################################
###############################################################################

if (__name__ == '__main__'):
    # Retrieve the paramaters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-strategy", default='PPO', type=str, help="Name of the trading strategy")
    parser.add_argument("-stock", default='Apple', type=str, help="Name of the stock (market)")
    parser.add_argument("-numberOfEpisodes", default=100, type=int, help="Number of training episodes")
    parser.add_argument("-batch_mode", default=False, type=bool, help="Batch Mode for training")
    parser.add_argument("-displayTestbench", default=False, type=bool, help="Dislay Testbench")
    parser.add_argument("-analyseTimeSeries", default=False, type=bool, help="Start Analysis Time Series")
    parser.add_argument("-simulateExistingStrategy", default=False, type=bool, help="Start Simulation of an Existing Strategy")
    parser.add_argument("-evaluateStrategy", default=False, type=bool, help="Start Evaluation of a Strategy")
    parser.add_argument("-evaluateStock", default=False, type=bool, help="Start Evaluation of a Stock")
    parser.add_argument("-crypto", default=False, type=bool, help="Start Evaluation of a Crypto Stock")
    args = parser.parse_args()

    # Initialization of the required variables
    simulator = TradingSimulator()
    strategy = args.strategy
    stock = args.stock
    multipleStock = False
    # check if stock are multiple, divided by char '-'
    if '-' in stock:
        stock = stock.split('-')
        print(stock)
        multipleStock = True

    numberOfEpisodes = args.numberOfEpisodes
    batch_mode = args.batch_mode
    displayTestbench = args.displayTestbench
    analyseTimeSeries = args.analyseTimeSeries
    simulateExistingStrategy = args.simulateExistingStrategy
    evaluateStrategy = args.evaluateStrategy
    evaluateStock = args.evaluateStock
    crypto = args.crypto

    startDate = startingDateCrypto if crypto else startingDate
    splitDate = splitingDateCrypto if crypto else splitingDate
    endDate = endingDateCrypto if crypto else endingDate

    # Training and testing of the trading strategy specified for the stock (market) specified
    if multipleStock:
        simulator.simulateMultipleStrategy(strategy, stock,
                                           startingDate=startDate, endingDate=endDate, splitingDate=splitDate,
                                           numberOfEpisodes=numberOfEpisodes, saveStrategy=False)
    else:
        simulator.simulateNewStrategy(strategy, stock,
                                      startingDate=startDate, endingDate=endDate, splitingDate=splitDate,
                                      numberOfEpisodes=numberOfEpisodes, batch_mode=batch_mode, saveStrategy=True)

    # the other functions can't be used with multipleStock, so it's used the first of the list
    if displayTestbench:
        simulator.displayTestbench()
    if analyseTimeSeries:
        simulator.analyseTimeSeries(stock[0] if multipleStock else stock,
                                    startingDate=startDate, endingDate=endDate, splitingDate=splitDate)
    if simulateExistingStrategy:
        simulator.simulateExistingStrategy(strategy, stock[0] if multipleStock else stock,
                                           startingDate=startDate, endingDate=endDate, splitingDate=splitDate)
    if evaluateStrategy:
        simulator.evaluateStrategy(strategy, saveStrategy=False,
                                   startingDate=startDate, endingDate=endDate, splitingDate=splitDate)
    if evaluateStock:
        simulator.evaluateStock(stock[0] if multipleStock else stock,
                                startingDate=startDate, endingDate=endDate, splitingDate=splitDate)


# Examples:
# Crypto Mode:      python3 main.py -stock Cardano -batch_mode True -crypto True
# Batch Mode:       python3 main.py -stock Tesla -batch_mode True
# Normal Call:      python3 main.py -stock Apple
# Multiple Assets:  python3 main.py -stock Tesla-Apple-Sony
