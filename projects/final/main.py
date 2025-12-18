import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
from time import time as clock
from plot import plot_agent_trajectory_with_cost
import random
from system import System


s = System()
print(s._get_cost(1.1))
s.plot_last_solution()
