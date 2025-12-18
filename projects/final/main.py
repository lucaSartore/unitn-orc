import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
from time import time as clock
from plot import plot_agent_trajectory_with_cost
import random
from system import SimpleSystem, System, InertiaSystem


# s = SimpleSystem()
s =  InertiaSystem()
print(s._get_solution(-1.1))
s.plot_last_solution()
