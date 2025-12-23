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

        bar = tqdm(range(3000))
        for i in bar:
            optim.zero_grad()
            self.critic.zero_grad()

            input = self.get_random_state(ACTOR_BATCH_SIZE)
            action: pt.Tensor = self.model(input)

            # state_cost = self.system.cost_function(input, action)
            # state_cost = action.T @ action
            input_cost = 0.5 * action * action
            x = input[:,0]

            position_cost = (x - 1.9) * (x - 1.0) * (x - 0.6) * (x + 0.5) * (x + 1.2) * (x + 2.1) 
            next_state = input + action * 0.02
            next_state_cost: pt.Tensor = self.critic(next_state)

            total_cost = next_state_cost + input_cost + position_cost

            loss = pt.sum(total_cost) / ACTOR_BATCH_SIZE

            loss.backward()
            
            optim.step()


            if i%10 == 0:
                with pt.no_grad():
                    optimal_cost = self.critic(input)
                    policy_cost = input_cost + self.critic(next_state)

                    optimality_loss = policy_cost - optimal_cost
                    optimality_loss = pt.sum(optimality_loss) / ACTOR_BATCH_SIZE

                    # print(optimality_loss.item())

                    bar.set_description(f"loss={optimality_loss.item():.2f}")

                # print(loss.item())
                # bar.set_description(f"loss={loss.item():.2f}")


            


class SimpleActor(Actor):
    def get_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.l1 = nn.Linear(1,250)
                self.l2 = nn.Linear(250,250)
                self.l3 = nn.Linear(250,250)
                self.l4 = nn.Linear(250,250)
                self.l5 = nn.Linear(250,1)

            def forward(self, x: pt.Tensor):
                x = self.l1(x)
                x = F.relu(x)
                x = self.l2(x)
                x = F.relu(x)
                x = self.l3(x)
                x = F.relu(x)
                x = self.l4(x)
                x = F.relu(x)
                x = self.l5(x)
                return x
        return Model()

    def get_random_state(self, batch_size: int) -> pt.Tensor:
        start, end = EXPLORATION_RANGE
        x = pt.rand((batch_size,1)) * (end-start) + start
        return x.to(DEVICE)


    def plot(self, system: System):
        x = np.linspace(*EXPLORATION_RANGE, 10)
        # Ensure model is in eval mode and move tensor to CPU for plotting
        self.model.eval()
        with pt.no_grad():
            y_critic = self.model(pt.tensor(x).float().unsqueeze(1).to(DEVICE)).cpu().numpy()
        
        # Convert the real solution list to a numpy array
        y_real = np.array([system.get_solution([i]).u_vector[0] for i in x])

        plt.figure(figsize=(10, 6))
        
        # Plotting the data
        plt.plot(x, y_real, label='Optimal policy', color='black', linestyle='--')
        plt.plot(x, y_critic, label='Actor policy', color='blue', alpha=0.8)
        
        # Formatting the chart
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
                self.l1 = nn.Linear(2,250)
                self.l2 = nn.Linear(250,250)
                self.l3 = nn.Linear(250,250)
                self.l4 = nn.Linear(250,250)
                self.l5 = nn.Linear(250,1)

            def forward(self, x: pt.Tensor):
                x = self.l1(x)
                x = F.relu(x)
                x = self.l2(x)
                x = F.relu(x)
                x = self.l3(x)
                x = F.relu(x)
                x = self.l4(x)
                x = F.relu(x)
                x = self.l5(x)
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
        x = np.linspace(*EXPLORATION_RANGE, 10)
        # Ensure model is in eval mode and move tensor to CPU for plotting
        self.model.eval()
        with pt.no_grad():
            y_critic = self.model(pt.tensor(x).float().unsqueeze(1).to(DEVICE)).cpu().numpy()
        
        # Convert the real solution list to a numpy array
        y_real = np.array([system.get_solution([i]).u_vector[0] for i in x])

        plt.figure(figsize=(10, 6))
        
        # Plotting the data
        plt.plot(x, y_real, label='Optimal policy', color='black', linestyle='--')
        plt.plot(x, y_critic, label='Actor policy', color='blue', alpha=0.8)
        
        # Formatting the chart
        plt.title('Comparison: Critic Model vs. Real System Solution')
        plt.xlabel('State Space ($x$)')
        plt.ylabel('Value ($V$)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.show()
