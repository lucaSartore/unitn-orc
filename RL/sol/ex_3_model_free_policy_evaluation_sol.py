#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:56:07 2021

@author: adelprete
"""
from copyreg import constructor
from matplotlib.typing import JoinStyleType
import numpy as np

from dpendulum import DPendulum

def mc_policy_eval(env: DPendulum, gamma, pi, nEpisodes, maxEpisodeLength, 
                   V_real, plot=False, nprint=1000):
    ''' Monte-Carlo Policy Evaluation:
        env: environment 
        gamma: discount factor
        pi: policy to evaluate
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        V_real: real Value table
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    # create a vector N to store the number of times each state has been visited
    N = np.zeros(env.nx)
    # create a vector C to store the cumulative cost associated to each state
    C = np.zeros(env.nx)
    # create a vector V to store the Value
    V = np.zeros(env.nx)
    # create a list V_err to store history of the error between real and estimated V table
    V_err = []
    
    # for each episode
    for k in range(nEpisodes):
        # reset the environment to a random state
        x = env.reset()

        x_list = []
        cost_list = []
        # keep track of the costs received at each state in this episode
        # simulate the system using the policy pi   
        for t in range(maxEpisodeLength):
            if callable(pi):
                u = pi(env, x) 
            else:
                u = pi[x]
            x = env.x
            x_list.append(x)

            x, cost = env.step(u)
            cost_list.append(cost)

        # Update the V-Table by computing the cost-to-go J backward in time        
        for t in range(maxEpisodeLength):
            J = 0
            for i in range(t, len(cost_list)):
                J += cost_list[i] * gamma ** (i - t)
            x = x_list[t]
            N[x] += 1
            C[x] += J
            V[x] = C[x] / N[x]

        # compute V_err as: mean(abs(V-V_real))
        V_err.append(np.mean(np.abs(V - V_real)))

        if k % nprint == 0:
            print(f"MC iter {k} V_err = {V_err[-1]}")
            if plot:
                env.plot_V_table(V)

    # V = np.zeros(env.nx);
    # V_err = []
    
    return V, V_err


def td0_policy_eval(env: DPendulum, gamma, pi, V0, nEpisodes, maxEpisodeLength, 
                    V_real, learningRate, plot=False, nprint=1000):
    ''' TD(0) Policy Evaluation:
        env: environment 
        gamma: discount factor
        pi: policy to evaluate
        V0: initial guess for V table
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        V_real: real Value table
        learningRate: learning rate of the algorithm
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    
    # make a copy of V0 using np.copy(V0)
    # create a list V__err to store the history of the error between real and estimated V table
    # for each episode
    # reset environment to random initial state
    # simulate the system using the policy pi
    # at each simulation step update the Value of the current state         
    # compute V_err as: mean(abs(V-V_real))

    # create a vector V to store the Value
    V = np.zeros(env.nx)
    # create a list V_err to store history of the error between real and estimated V table
    V_err = []
    
    # for each episode
    for k in range(nEpisodes):
        # reset the environment to a random state
        x = env.reset()

        # keep track of the costs received at each state in this episode
        # simulate the system using the policy pi   
        for _ in range(maxEpisodeLength):
            x = env.x
            if callable(pi):
                u = pi(env, x) 
            else:
                u = pi[x]
            x_new, cost = env.step(u)
            V[x] = (1-learningRate) * V[x] + learningRate * (cost + gamma * V[x_new])

        # compute V_err as: mean(abs(V-V_real))
        V_err.append(np.mean(np.abs(V - V_real)))

        if k % nprint == 0:
            print(f"TD iter {k} V_err = {V_err[-1]}")
            if plot:
                env.plot_V_table(V)

    return V, V_err

