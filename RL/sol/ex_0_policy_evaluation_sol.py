#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 05:30:56 2021

@author: adelprete
"""
import numpy as np

from RL.dpendulum import DPendulum

def policy_eval(env: DPendulum, gamma, pi, V, maxIters, threshold, plot=False, nprint=1000):
    ''' Policy evaluation algorithm 
        env: environment used for evaluating the policy
        gamma: discount factor
        pi: policy to evaluate
        V: initial guess of the Value table
        maxIters: max number of iterations of the algorithm
        threshold: convergence threshold
        plot: if True it plots the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    
    # IMPLEMENT POLICY EVALUATION HERE
    
    # Iterate for at most maxIters loops
    for i in range(maxIters):
        v_old = np.copy(V)
        for state in range(env.nx):

            # Use env.reset(x) to set the robot state
            env.reset(state)
            # To simulate he system use env.step(u) which returns the next state and the cost
            if callable(pi):
                action = pi(env, state)
            else:
                action = pi[state]
            nextState, cost = env.step(action)

            # Update V-Table with Bellman's equation
            V[state] = cost + gamma * v_old[nextState]

        # Check for convergence using the difference between the current and previous V table
        if np.max(np.abs(V - v_old)) < threshold:
            break

        if i % nprint == 0 and plot:
            env.plot_V_table(V)

    return V
