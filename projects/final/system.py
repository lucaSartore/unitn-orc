from abc import ABC
import numpy as np
import casadi as cs
from plot import plot_agent_trajectory_with_cost
from dataclasses import dataclass
import random

from numpy.typing import NDArray

@dataclass
class Solution:
    x_vector: NDArray[np.float64]
    u_vector: NDArray[np.float64]
    score: float


# class System(ABC):
class System:

    def __init__(
        self,
        s_initialization_range: tuple[float, float] = (-2.1,1.9),
        dt: float = 0.02,
        horizon: int = 150
    ):
        # random initialization range for the state variable
        self.s_initialization_range = s_initialization_range
        # time step size
        self.dt = dt
        # horizon size
        self.horizon = horizon
        # number of dimensions of the state variable x
        self.nx = 1
        # number of dimensions of the control variable u
        self.nu = 1 

        # last solution
        self.last_solution: Solution | None = None

    def cost_function(self, u, x):
        return 0.5 * u * u + (x - 1.9) * (x - 1.0) * (x - 0.6) * (x + 0.5) * (x + 1.2) * (x + 2.1) 

    def plot_last_solution(self):
        assert self.last_solution != None, "solution has not being correctly saved"

        timestamps = np.linspace(0,(self.horizon+1) * self.dt, self.horizon+1)
        plot_agent_trajectory_with_cost(
            timestamps,
            [self.last_solution.x_vector],
            lambda x: self.cost_function(0,x)
        )

    def _get_cost(self, initial_position: float) -> float:

        # state variable s = [x] #position
        s   = cs.SX.sym('s', self.nx)
        # control variable u = [u] #velocity
        u   = cs.SX.sym('u', self.nu)

        # state transition function
        f = cs.Function('f', [u], [u])

        # optimizer
        opti = cs.Opti()

        # initial value of the state s
        param_s_zero = opti.parameter(self.nx)

        # creating the variables for each time step
        X, U = [], []
        for k in range(self.horizon+1): 
            x = opti.variable(self.nx)
            opti.set_initial(x, random.uniform(*self.s_initialization_range))
            X += [x]
        for k in range(self.horizon): 
            U += [opti.variable(self.nu)]

        #accumulated cost value
        cost = 0


        def cost_function(u,x):
            return 0.5 * u * u + (x - 1.9) * (x - 1.0) * (x - 0.6) * (x + 0.5) * (x + 1.2) * (x + 2.1) 
            # return (x - 1.9) * (x - 1.0) * (x - 0.6) * (x + 0.5) * (x + 1.2) * (x + 2.1) 

        # adding constraints
        opti.subject_to(X[0] == param_s_zero)
        for k in range(self.horizon):     
            u = U[k]
            x = X[k]
            cost += cost_function(u,x)
            opti.subject_to(X[k+1] == x + self.dt * f(u)) #type: ignore

        opti.minimize(cost)

        opts = {
            "ipopt.print_level": 1,
            "ipopt.tol": 1e-6,
            "ipopt.constr_viol_tol": 1e-6,
            "ipopt.compl_inf_tol": 1e-6,
            "print_time": 0,
            "detect_simple_bounds": True
        }
        opti.solver("ipopt", opts)

        opti.set_value(param_s_zero, [-1.3])
        sol = opti.solve()

        x_sol = np.array([sol.value(X[k]) for k in range(self.horizon+1)], dtype=np.float64)
        u_sol = np.array([sol.value(U[k]) for k in range(self.horizon)], dtype=np.float64)
        score = float(sol.value(cost))
        
        self.last_solution = Solution(x_sol, u_sol, score)
        return score

