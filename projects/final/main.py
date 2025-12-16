import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
from time import time as clock
from plot import plot_agent_trajectory_with_cost


# time step
dt = 0.02

# time horizon
N = 150

# number of dimensions of the state variable x
nx = 1

# number of dimensions of the control variable u
nu = 1 


# state variable s = [x] #position
s   = cs.SX.sym('s', nx)
# control variable u = [u] #velocity
u   = cs.SX.sym('u', nu)

# state transition function
f = cs.Function('f', [u], [u])

# optimizer
opti = cs.Opti()

# initial value of the state s
param_s_zero = opti.parameter(nx)

# creating the variables for each time step
X, U = [], []
for k in range(N+1): 
    X += [opti.variable(nx)]
for k in range(N): 
    U += [opti.variable(nu)]

#accumulated cost value
cost = 0


def cost_function(u,x):
    return 0.5 * u ** 2 + (x - 1.9) * (x - 1.0) * (x - 0.6) * (x + 0.5) * (x + 1.2) * (x + 2.1) 
    # return (x - 1.9) * (x - 1.0) * (x - 0.6) * (x + 0.5) * (x + 1.2) * (x + 2.1) 

# adding constraints
opti.subject_to(X[0] == param_s_zero)
for k in range(N):     
    u = U[k]
    x = X[k]
    cost += cost_function(u,x)
    opti.subject_to(X[k+1] == x + dt * f(u)) #type: ignore
    opti.subject_to(X[k+1] == x + dt * f(u)) #type: ignore

opti.minimize(cost)

opts = {
    # "ipopt.print_level": 1,
    # "ipopt.tol": 1e-6,
    # "ipopt.constr_viol_tol": 1e-6,
    # "ipopt.compl_inf_tol": 1e-6,
    # "print_time": 0,
    # "detect_simple_bounds": True
}
opti.solver("ipopt", opts)

opti.set_value(param_s_zero, [-2.15])
sol = opti.solve()
x_sol = np.array([sol.value(X[k]) for k in range(N+1)]).T
u_sol = np.array([sol.value(U[k]) for k in range(N)]).T

timestamps = np.linspace(0,(N+1) * dt, N+1)
plot_agent_trajectory_with_cost(
    timestamps,
    [x_sol],
    lambda x: cost_function(0,x)
)

print(x_sol)
print(u_sol)

