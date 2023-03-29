import time
import os
import errno

import pandas as pd
import numpy as np
import traceback

from agent import DRLAgent
from meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from meta.preprocessor.preprocessors import data_split

def prepare_rolling_train(df, date_column, testing_window, max_rolling_window, trade_date):
    print(trade_date-max_rolling_window, trade_date-testing_window)
    train = data_split(df, trade_date - max_rolling_window, trade_date - testing_window)
    #print(train)
    return train

def prepare_rolling_test(df, date_column, testing_window, max_rolling_window, trade_date):
    test = data_split(df, trade_date-testing_window, trade_date)
    X_test = test.reset_index()
    return test

def train_ppo(agent):
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.005,
        "learning_rate": 0.0001,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
    trained_ppo = agent.train_model(model=model_ppo, tb_log_name='ppo', total_timesteps=80000)
    return trained_ppo


def run_ppo_model(df, date_column, trade_date, env_kwargs, 
                  start_date, end_date, split_date,
                  testing_window=4, max_rolling_window=44):
    ## initialize all the result tables
    ## need date as index and unique tic name as columns
    # first trade date is 1995-06-01
    # fist_trade_date_index = 20
    # testing_windows = 6
    X_train = data_split(df, start_date, split_date)
    # prepare_rolling_train(df, date_column, testing_window, max_rolling_window, trade_date)
    # prepare testing data
    X_test = data_split(df, split_date, end_date)
    # prepare_rolling_test(df, date_column, testing_window, max_rolling_window, trade_date)
    print(X_train)
    e_train_gym = StockPortfolioEnv(df=X_train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env = env_train)
    ppo_model = train_ppo(agent)
    
    # Test
    e_trade_gym = StockPortfolioEnv(df=X_test, **env_kwargs)    
    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=ppo_model, environment=e_trade_gym)
    ppo_return = list((df_daily_return.daily_return+1).cumprod())[-1] 
    return ppo_model, ppo_return