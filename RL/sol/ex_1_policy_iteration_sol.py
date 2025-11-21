#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 05:30:56 2021

@author: adelprete
"""
import numpy as np
#from sol.ex_0_policy_evaluation_prof import policy_eval
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from RL.dpendulum import DPendulum
else:
    DPendulum = Any
from ex_0_policy_evaluation import policy_eval

def policy_iteration(env: DPendulum, gamma, pi, V, maxEvalIters, maxImprIters, value_thr, policy_thr, plot=False, nprint=1000):
    ''' Policy iteration algorithm 
        env: environment used for evaluating the policy
        gamma: discount factor
        pi: initial guess for the policy (array with size nev.nx containing the discrete integer action)
        V: initial guess of the Value table
        maxEvalIters: max number of iterations for policy evaluation
        maxImprIters: max number of iterations for policy improvement
        value_thr: convergence threshold for policy evaluation
        policy_thr: convergence threshold for policy improvement
        plot: if True it plots the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    # IMPLEMENT POLICY ITERATION HERE
    
    # Create an array to store the Q value of different controls
    Q = np.ones(shape=(env.nx, env.nu))

    # Iterate at most maxImprIters loops
    initial_v = np.copy(V)
    for i in range(maxImprIters):
        print(f"iteration {i}")
        # Make a copy of current policy table
        pi_old = np.copy(pi)

        # Evaluate current policy using policy_eval for at most maxEvalIters iterations 
        V = policy_eval(env, gamma, pi, initial_v, maxEvalIters, value_thr)


        for state in range(env.nx):
            for control in range(env.nu):
                env.reset(state)
                x_next,cost = env.step(control)
                Q[state,control] = cost + gamma * V[x_next]


        pi = np.argmin(Q,1)

        if all(pi == pi_old):
            break

        if i % nprint == 0 and plot:
            env.plot_V_table(V)
            env.plot_policy(pi)

    return pi
