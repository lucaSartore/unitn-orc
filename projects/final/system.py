from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import casadi as cs
from plot import plot_agent_trajectory_with_cost
from dataclasses import dataclass
import random

from numpy.typing import NDArray

@dataclass
class Solution:
    pos_vector: NDArray[np.float64]
    u_vector: NDArray[np.float64]
    score: float


# class System(ABC):
class System:

    def __init__(
        self,
        nx: int,
        nu: int,
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
        self.nx = nx
        # number of dimensions of the control variable u
        self.nu = nu

        # last solution
        self.last_solution: Solution | None = None

    def cost_function(self, x, u):
        return 0.5 * u * u + (x - 1.9) * (x - 1.0) * (x - 0.6) * (x + 0.5) * (x + 1.2) * (x + 2.1) 

    def plot_last_solution(self):
        assert self.last_solution != None, "solution has not being correctly saved"

        timestamps = np.linspace(0,(self.horizon+1) * self.dt, self.horizon+1)
        plot_agent_trajectory_with_cost(
            timestamps,
            [self.last_solution.pos_vector],
            lambda x: self.cost_function(x,0)
        )

    @abstractmethod
    def state_transition_function(self, s,u) -> list[Any]:
        """
        function that calculate the state difference between
        the current and the next.

        return a list of any, because s and u can be real number
        or casadi variables
        """
        pass

    def _get_solution(self, initial_position: float) -> Solution:

        # state variable s = [x] #position
        s   = cs.SX.sym('s', self.nx)
        # control variable u = [u] #velocity
        u   = cs.SX.sym('u', self.nu)

        # state transition function
        rhs = cs.vertcat(*self.state_transition_function(s,u))
        f = cs.Function('f', [s,u], [rhs])

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

        # adding constraints
        opti.subject_to(X[0] == param_s_zero)
        for k in range(self.horizon):     
            u = U[k]
            x = X[k]
            cost += self.cost_function(x,u)
            opti.subject_to(X[k+1] == x + f(x,u))

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

        opti.set_value(param_s_zero, [initial_position])
        sol = opti.solve()

        x_sol = np.array([sol.value(X[k]) for k in range(self.horizon+1)], dtype=np.float64)
        u_sol = np.array([sol.value(U[k]) for k in range(self.horizon)], dtype=np.float64)
        score = float(sol.value(cost))
        
        self.last_solution = Solution(x_sol, u_sol, score)
        return self.last_solution


class SimpleSystem(System):

    def __init__(self, s_initialization_range: tuple[float, float] = (-2.1, 1.9), dt: float = 0.02, horizon: int = 150):
        super().__init__(1, 1, s_initialization_range, dt, horizon)

    def state_transition_function(self, s, u):
        return [u * self.dt]



class InertiaSystem(System):

    def __init__(self, s_initialization_range: tuple[float, float] = (-2.1, 1.9), dt: float = 0.02, horizon: int = 150):
        super().__init__(2, 1, s_initialization_range, dt, horizon)

    def cost_function(self, x, u):
        # need to pass only the position (not the velocity)
        if type(x) == cs.MX:
            x = x[0]
        return super().cost_function(x,u)

    def state_transition_function(self, s, u):
        v = s[1]
        return [v*self.dt + 0.5 * u * self.dt**2, u*self.dt]

    def _get_solution(self, initial_position: float) -> Solution:
        # intercept the call, to save the solution with just the position vector
        # (and remove the velocity part of the acceleration)
        sol = super()._get_solution(initial_position)
        sol.pos_vector = sol.pos_vector[:,0]
        print(sol.pos_vector.shape)
        self.last_solution = sol
        return sol

