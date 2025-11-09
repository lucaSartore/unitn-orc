#!/usr/bin/env python3
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import casadi as cs
import time
from time import sleep

from adam.casadi.computations import KinDynComputations
from example_robot_data.robots_loader import load
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.viz_utils import addViewerSphere, applyViewerConfiguration
from pinocchio.utils import rpyToMatrix


# ====================== Simple plotting utility ======================
def plot_infinity(t_init, t_final):
    r = 0.1
    t = np.linspace(t_init, t_final, 300)
    x = r * np.cos(2*np.pi*t)
    y = r * 0.5*np.sin(4*np.pi*t)
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, 'x')
    plt.xlim([-0.11, 0.11])
    plt.ylim([-0.06, 0.06])
    plt.title("Infinity-shaped path")
    plt.grid(True)
    plt.show()


# ====================== Robot and Dynamics Setup ======================
robot = load("ur5")
joints_name_list = [s for s in robot.model.names[1:]]
nq = len(joints_name_list)
nx = 2 * nq

dt = 0.02
N = 100
q0 = np.zeros(nq)
dq0 = np.zeros(nq)
x_init = np.concatenate([q0, dq0])
w_max = 5
frame_name = "ee_link"
r_path = 0.2

# CasADi symbolic variables
q = cs.SX.sym('q', nq)
dq = cs.SX.sym('dq', nq)
ddq = cs.SX.sym('ddq', nq)
state = cs.vertcat(q, dq)
rhs = cs.vertcat(dq, ddq)
f = cs.Function('f', [state, ddq], [rhs])

# Inverse dynamics
kinDyn = KinDynComputations(robot.urdf, joints_name_list)
H_b = cs.SX.eye(4)
v_b = cs.SX.zeros(6)
bias_forces = kinDyn.bias_force_fun()
mass_matrix = kinDyn.mass_matrix_fun()
h = bias_forces(H_b, q, v_b, dq)[6:]
M = mass_matrix(H_b, q)[6:, 6:]
tau = M @ ddq + h
inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

# Forward kinematics
fk_fun = kinDyn.forward_kinematics_fun(frame_name)
ee_pos = fk_fun(H_b, q)[:3, 3]
fk = cs.Function('fk', [q], [ee_pos])

y = fk(q0)
c_path = np.array([y[0]-r_path, y[1], y[2]]).squeeze()

lbx = robot.model.lowerPositionLimit.tolist() + (-robot.model.velocityLimit).tolist()
ubx = robot.model.upperPositionLimit.tolist() + robot.model.velocityLimit.tolist()
tau_min = (-robot.model.effortLimit).tolist()
tau_max = robot.model.effortLimit.tolist()


# ====================== Optimization Problem ======================
def create_decision_variables(N, nx: int, nu, lbx, ubx):
    opti = cs.Opti()
    X, U, S, W = [], [], [], []
    for _ in range(N + 1):
        X += [opti.variable(nx)]
        opti.subject_to(opti.bounded(lbx, X[-1], ubx))
        S += [opti.variable(1)]
        opti.subject_to(opti.bounded(0, S[-1], 1))
    for _ in range(N):
        U += [opti.variable(nu)]
        W += [opti.variable(1)]
    dt = opti.variable(1)
    return opti, X, U, S, W, dt


# squared norm function
squared_norm = lambda x: x.T @ x

# notes:
# NJ = number of joints
# N = number of time-steps
# X contains for each time step T the system status X = [q .. qd] shape = [N+1,2*NJ]
# S contains for each time step T the decision variable s (that is strictly increasing) shape = [N+s, 1]
# U is the control input (aka joint accelerations) shape = [N, NJ]
# W is the incremental time step. S[i+1] = S[i] + dt * W[i] shape = [N, 1]
# c_path is the center of the path's circle
# r_path is the radius of the path's circle
# w_v, w_a and w_w are the weights of the tasks to minimize (velocity, acceleration and control input)

def define_running_cost_and_dynamics(opti: cs.Opti, X, U, S, W, N, dt, x_init,
                                     c_path, r_path, w_v, w_a, w_w, w_dt,
                                     tau_min, tau_max):
    
    # TODO: Constrain the initial state X[0] to be equal to the initial condition x_init
    opti.subject_to(X[0] == x_init)
    

    # TODO: Initialize the path variable S[0] to 0.0
    opti.subject_to(S[0] == 0.0)
    

    # TODO: Constrain the final path variable S[-1] to be 1.0
    opti.subject_to(S[-1] == 1.0)

    
    # constraints on the dt boundry
    opti.subject_to(dt >= 0.001)
    opti.subject_to(dt <= 0.1)

    cost = 0.0

    for k in range(N):

        # running cost attributed to time step
        cost += w_dt * (dt ** 2)

        qk = X[k][:nq]
        qdk = X[k][nq:]

        # TODO: Compute the end-effector position using forward kinematics
        ee_pos = fk(qk)


        ee_des = np.array([c_path[0] + r_path*np.cos(2*np.pi*S[k]),
                             c_path[1] + r_path*0.5*np.sin(4*np.pi*S[k]),
                             c_path[2]])

        # TODO: Constrain ee_pos to lie on the desired path in x, y, z
        opti.subject_to(ee_des[0] == ee_pos[0])
        opti.subject_to(ee_des[1] == ee_pos[1])
        opti.subject_to(ee_des[2] == ee_pos[2])


        # TODO: Add velocity tracking cost term
        cost += w_v * squared_norm(qdk)

        # TODO: Add actuation effort cost term
        cost += w_a * squared_norm(U[k])

        # TODO: Add path progression speed cost term
        cost += w_w * W[k]**2

        opti.minimize(cost)

        # TODO: Add discrete-time dynamics constraint

        # q and qd at next time step
        qk2 = X[k+1][:nq]
        qdk2 = X[k+1][nq:]

        opti.subject_to(qk2 == qk + dt * qdk)
        opti.subject_to(qdk2 == qdk + dt * U[k])
        

        # TODO: Add path variable dynamics constraint
        opti.subject_to(S[k+1] == S[k] + W[k] * dt)
        opti.subject_to(W[k] > 0)
        

        # TODO: Constrain the joint torques to remain within [tau_min, tau_max]
        tau = inv_dyn(X[k], U[k])
        opti.subject_to(tau_min <= tau)
        opti.subject_to(tau <= tau_max)
        
        
    return cost

def define_terminal_cost_and_constraints(opti, X, S, c_path, r_path, w_final):
    # TODO: Compute the end-effector position at the final state
    q_last = X[-1][:nq]
    ee_pos = fk(q_last)
    

    # TODO: Constrain ee_pos to lie on the desired path in x, y, z at the end
    
    # note: we could simplify the expression as S[-1] must be 1, but I think this is more readable
    ee_des = np.array([c_path[0] + r_path*np.cos(2*np.pi*S[-1]),
                         c_path[1] + r_path*0.5*np.sin(4*np.pi*S[-1]),
                         c_path[2]])

    opti.subject_to(ee_des[0] == ee_pos[0])
    opti.subject_to(ee_des[1] == ee_pos[1])
    opti.subject_to(ee_des[2] == ee_pos[2])

    cost = w_final * squared_norm(X[0] - X[-1])
    return cost



def create_and_solve_ocp(N, nx, nq, lbx, ubx, dt, x_init,
                         c_path, r_path, w_v, w_a, w_w, w_dt, w_final,
                         tau_min, tau_max):
    opti, X, U, S, W, dt = create_decision_variables(N, nx, nq, lbx, ubx)
    running_cost = define_running_cost_and_dynamics(opti, X, U, S, W, N, dt, x_init,
                                                    c_path, r_path, w_v, w_a, w_w, w_dt,
                                                    tau_min, tau_max)
    terminal_cost = define_terminal_cost_and_constraints(opti, X, S, c_path, r_path, w_final)
    opti.minimize(running_cost + terminal_cost)

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.tol": 1e-4,
        "ipopt.hessian_approximation":"limited-memory",
    }

    opti.solver("ipopt", opts)

    t0 = time.time()
    sol = opti.solve()
    print(f"Solver time: {time.time() - t0:.2f}s")
    return sol, X, U, S, W, dt


def extract_solution(sol, X, U, S, W, dt):
    x_sol = np.array([sol.value(X[k]) for k in range(N + 1)]).T
    ddq_sol = np.array([sol.value(U[k]) for k in range(N)]).T
    s_sol = np.array([sol.value(S[k]) for k in range(N + 1)]).T
    q_sol = x_sol[:nq, :]
    dq_sol = x_sol[nq:, :]
    w_sol = np.array([sol.value(W[k]) for k in range(N)]).T
    tau = np.zeros((nq, N))
    for i in range(N):
        tau[:, i] = inv_dyn(x_sol[:, i], ddq_sol[:, i]).toarray().squeeze()
    ee = np.zeros((3, N + 1))
    for i in range(N + 1):
        ee[:, i] = fk(x_sol[:nq, i]).toarray().squeeze()
    ee_des = np.zeros((3, N + 1))
    for i in range(N + 1):
        ee_des[:, i] = np.array([c_path[0] + r_path*np.cos(2*np.pi*s_sol[i]),
                                 c_path[1] + r_path*0.5*np.sin(4*np.pi*s_sol[i]),
                                 c_path[2]])
    dt_sol = sol.value(dt)
    return q_sol, dq_sol, ddq_sol, tau, ee, ee_des, s_sol, w_sol, dt_sol


# ====================== Simulation and Visualization ======================
r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
simu = RobotSimulator(conf_ur5, r)
simu.init(q0, dq0)
simu.display(q0)

REF_SPHERE_RADIUS = 0.02
EE_REF_SPHERE_COLOR = np.array([1, 0, 0, .5])


def display_motion(q_traj, ee_des_traj):
    for i in range(N + 1):
        t0 = time.time()
        simu.display(q_traj[:, i])
        addViewerSphere(r.viz, f'world/ee_ref_{i}', REF_SPHERE_RADIUS, EE_REF_SPHERE_COLOR)
        applyViewerConfiguration(r.viz, f'world/ee_ref_{i}', ee_des_traj[:, i].tolist() + [0, 0, 0, 1.])
        t1 = time.time()
        if(t1-t0 < dt):
            sleep(dt - (t1-t0))



# ====================== Main Execution ======================
if __name__ == "__main__":
    print("Plotting reference infinity curve...")
    plot_infinity(0, 1)

    log_w_v, log_w_a, log_w_w, log_w_final = -6, -6, -6, -6
    log_w_p = 2 #Log of trajectory tracking cost 
    log_w_dt = -1 #Log of running cost

    sol, X, U, S, W, dt_opt = create_and_solve_ocp(
        N, nx, nq, lbx, ubx, dt, x_init, c_path, r_path,
        10**log_w_v, 10**log_w_a, 10**log_w_w, 10 ** log_w_dt, 10**log_w_final, 
        tau_min, tau_max
    )
    q_sol, dq_sol, u_sol, tau, ee, ee_des, s_sol, w_sol, dt_sol = extract_solution(sol, X, U, S, W, dt_opt)

    print("Displaying robot motion...")

    print(f"final time-steps size: {dt_sol}")

    for i in range(3):
        display_motion(q_sol, ee_des)

    # Plot results
    tt = np.linspace(0, (N + 1) * dt, N + 1)
    plt.figure(figsize=(10, 4))
    plt.plot([tt[0], tt[-1]], [0, 1], ':', label='straight line', alpha=0.7)
    plt.plot(tt, s_sol, label='s')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(ee_des[0,:].T, ee_des[1,:].T, 'r x', label='EE des', alpha=0.7)
    plt.plot(ee[0,:].T, ee[1,:].T, 'k o', label='EE', alpha=0.7)
    plt.xlabel('End-effector pos x [m]')
    plt.ylabel('End-effector pos y [m]')
    plt.legend()
    plt.grid(True)
    
    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.plot(tt, ee_des[i,:].T, ':', label=f'EE des {i}', alpha=0.7)
        plt.plot(tt, ee[i,:].T, label=f'EE {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('End-effector pos [m]')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    for i in range(dq_sol.shape[0]):
        plt.plot(tt, dq_sol[i,:].T, label=f'dq {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint velocity [rad/s]')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    for i in range(q_sol.shape[0]):
        plt.plot([tt[0], tt[-1]], [q_sol[i,0], q_sol[i,0]], ':', label='straight line', alpha=0.7)
        plt.plot(tt, q_sol[i,:].T, label=f'q {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint [rad]')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    for i in range(tau.shape[0]):
        plt.plot(tt[:-1], tau[i,:].T, label=f'tau {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint torque [Nm]')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    plt.plot(tt[:-1], w_sol.T, label=f'w', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid(True)
    plt.show()
