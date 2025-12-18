import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
from time import time as clock
from plot import plot_agent_trajectory_with_cost
import random
from system import GreedyPolicy, SimpleSystem, System, InertiaSystem


# s = SimpleSystem()
# print(s._get_solution([-1.1]))
# s.evaluate_policy(GreedyPolicy(), [1.5])

s =  InertiaSystem()
print(s._get_solution([-1.1, 0]))
print(s.evaluate_policy(GreedyPolicy(), [1.5,0]))

s.plot_last_solution()
