from abc import ABC, abstractmethod
import math
from os import stat
from matplotlib.colors import AsinhNorm
import torch as pt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from critic import Critic
import critic
from dataset import RobotDataset
from parameters import ACTOR_BATCH_SIZE, ACTOR_MAX_TRAINING_GENERATIONS, CRITIC_BATCH_SIZE, DEVICE, CRITIC_LEARNING_RATE, CRITIC_MAX_TRAINING_GENERATIONS, EXPLORATION_RANGE, PATIENCE, TRAIN_TEST_SPLIT, VELOCITY_RANGE
from copy import copy, deepcopy
from system import Policy, System
import numpy as np
import matplotlib.pyplot as plt



class Actor(ABC):

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    @abstractmethod
    def plot(self, system: System):
        pass

    @abstractmethod
    def get_random_state(self, batch_size: int) -> pt.Tensor:
        pass

    def __init__(self, system: System, critic: Critic) -> None:
        self.system = system
        self.critic = critic.model
        self.model = self.get_model().to(DEVICE)

    def run(self):
        self.train()

    def get_policy(self) -> Policy:
        class ActorBasedPolicy(Policy):
            def __init__(self, model: nn.Module):
                self.model = deepcopy(model).to(DEVICE)

            def run(self, x) -> float:
                input = pt.tensor(x, dtype=pt.float32).to(DEVICE)
                output: pt.Tensor = self.model(input)
                return float(output.item())

        return ActorBasedPolicy(self.model)

    def train(self):
        self.critic = self.critic.eval()
        self.model = self.model.train()

        for param in self.critic.parameters():
            param.requires_grad = False

        optim = pt.optim.AdamW(self.model.parameters(), lr=0.00005)

        bar = tqdm(range(ACTOR_MAX_TRAINING_GENERATIONS))
        for i in bar:
            optim.zero_grad()
            self.critic.zero_grad()

            input = self.get_random_state(ACTOR_BATCH_SIZE)
            action: pt.Tensor = self.model(input)

            state_cost = self.system.cost_function(input, action)

            next_state = input + pt.hstack(self.system.state_transition_function(input, action)).to(DEVICE)
            next_state_cost: pt.Tensor = self.critic(next_state)

            total_cost = next_state_cost + state_cost

            loss = pt.sum(total_cost) / ACTOR_BATCH_SIZE

            loss.backward()
            
            optim.step()

            if i%10 == 0:
                bar.set_description(f"loss={loss.item():.2f}")


            


class SimpleActor(Actor):
    def get_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.l1 = nn.Linear(1,512)
                self.l2 = nn.Linear(512,128)
                self.l3 = nn.Linear(128,1)

            def forward(self, x: pt.Tensor):
                x = self.l1(x)
                x = F.leaky_relu(x)
                x = self.l2(x)
                x = F.leaky_relu(x)
                x = self.l3(x)
                return x
        return Model()

    def get_random_state(self, batch_size: int) -> pt.Tensor:
        start, end = EXPLORATION_RANGE
        x = pt.rand((batch_size,1)) * (end-start) + start
        return x.to(DEVICE)


    def plot(self, system: System):
        x = np.linspace(*EXPLORATION_RANGE, 50)
        self.model.eval()
        with pt.no_grad():
            y_critic = self.model(pt.tensor(x).float().unsqueeze(1).to(DEVICE)).cpu().numpy()
        
        y_real = np.array([system.get_solution([i]).u_vector[0] for i in x])

        plt.figure(figsize=(10, 6))
        
        plt.plot(x, y_real, label='Optimal policy', color='black', linestyle='--')
        plt.plot(x, y_critic, label='Actor policy', color='blue', alpha=0.8)
        plt.title('Comparison: Critic Model vs. Real System Solution')
        plt.xlabel('State Space ($x$)')
        plt.ylabel('Value ($V$)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.show()

class InertiaActor(Actor):
    def get_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.l1 = nn.Linear(2,1024)
                self.l2 = nn.Linear(1024,256)
                self.l3 = nn.Linear(256,64)
                self.l4 = nn.Linear(64,1)

            def forward(self, x: pt.Tensor):
                x = self.l1(x)
                x = F.leaky_relu(x)
                x = self.l2(x)
                x = F.leaky_relu(x)
                x = self.l3(x)
                x = F.leaky_relu(x)
                x = self.l4(x)
                return x
        return Model()

    def get_random_state(self, batch_size: int) -> pt.Tensor:
        start_x, end_x = EXPLORATION_RANGE
        start_v, end_v = VELOCITY_RANGE
        x = pt.rand((batch_size,1)) * (end_x-start_x) + start_x
        v = pt.rand((batch_size,1)) * (end_v-start_v) + start_v
        state = pt.hstack([x,v])
        return state.to(DEVICE)



    def plot(self, system: System):
        SPACE_POINTS = 10
        VELOCITY_POINTS = 10
        x = np.linspace(*EXPLORATION_RANGE, SPACE_POINTS)
        v = np.linspace(*VELOCITY_RANGE, VELOCITY_POINTS)

        plt.style.use('_mpl-gallery')
        x, v = np.meshgrid(x, v)


        self.model.eval().to(DEVICE)
        with pt.no_grad():
            xt = pt.tensor(x).float().to(DEVICE).reshape((SPACE_POINTS * VELOCITY_POINTS, 1))
            vt = pt.tensor(v).float().to(DEVICE).reshape((SPACE_POINTS * VELOCITY_POINTS, 1))
            input = pt.hstack([xt, vt]).to(DEVICE)
            y_critic: pt.Tensor = self.model(input)
            y_critic = y_critic.reshape((SPACE_POINTS, VELOCITY_POINTS))
            y_critic_numpy = y_critic.detach().cpu().numpy()

        y_gt = np.zeros(shape=x.shape, dtype=np.float32)
        for i in range(SPACE_POINTS):
            for j in range(VELOCITY_POINTS):
                state = [float(x[i,j]), float(v[i,j])]
                y_gt[i,j] = system.get_solution(state).u_vector[0]

        
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(x, v, y_critic_numpy, color='blue', alpha=0.5, label='Actor policy') #type: ignore
        ax.plot_surface(x, v, y_gt, color='red', alpha=0.5, label='Optimal policy') #type: ignore

        ax.legend()
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Control Input') #type: ignore

        plt.show()
