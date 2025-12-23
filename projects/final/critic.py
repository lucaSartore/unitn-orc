from abc import ABC, abstractmethod
import math
import torch as pt
from torch import detach, nn
from torch._prims_common import dtype_or_default
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataset import RobotDataset
from parameters import CRITIC_BATCH_SIZE, DEVICE, CRITIC_LEARNING_RATE, CRITIC_MAX_TRAINING_GENERATIONS, EXPLORATION_RANGE, PATIENCE, TRAIN_TEST_SPLIT, VELOCITY_RANGE
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from system import System


class Critic(ABC):

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    @abstractmethod
    def plot(self, system: System):
        pass

    def __init__(self, dataset: RobotDataset) -> None:
        self.dataset = dataset
        self.model = self.get_model()

    def run(self):
        train_set, val_set = pt.utils.data.random_split(self.dataset, TRAIN_TEST_SPLIT)
        self.train(train_set)
        self.validate(val_set)

    def validate(self, dataset: Dataset):
        data_loader = DataLoader(
            dataset,
            batch_size=CRITIC_BATCH_SIZE,
            shuffle=True
        )

        self.model.eval()
        self.model = self.model.to(DEVICE)

        running_loss = 0
        input: pt.Tensor
        output: pt.Tensor
        for input,output in data_loader:
            input = input.to(DEVICE)
            if len(input.shape) == 1:
                input = input.unsqueeze(1)
            output = output.to(DEVICE)
            output = output.unsqueeze(1)

            predicted_output: pt.Tensor = self.model(input)
            loss = F.l1_loss(predicted_output, output)

            running_loss += loss.item()

        final_loss = running_loss / len(data_loader)

        print(f"Final loss with validation dataset: {final_loss}")
                 

    def train(self, dataset: Dataset):
        
        data_loader = DataLoader(
            dataset,
            batch_size=CRITIC_BATCH_SIZE,
            shuffle=True
        )

        self.model.train()
        self.model = self.model.to(DEVICE)

        optimizer = pt.optim.AdamW(params=self.model.parameters(), lr=CRITIC_LEARNING_RATE)

        scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=10,
            threshold=1e-4
        )

        
        best_model: nn.Module | None = None
        patience = PATIENCE
        best_loss = math.inf
        bar = tqdm(range(CRITIC_MAX_TRAINING_GENERATIONS))
        for _ in bar:

            running_loss = 0
            input: pt.Tensor
            output: pt.Tensor
            for input,output in data_loader:
                input = input.to(DEVICE)
                if len(input.shape) == 1:
                    input = input.unsqueeze(1)
                output = output.to(DEVICE)
                output = output.unsqueeze(1)
                optimizer.zero_grad()

                predicted_output: pt.Tensor = self.model(input)
                loss = F.l1_loss(predicted_output, output)
                loss.backward()

                optimizer.step()
                
                running_loss += loss.item()

            final_loss = running_loss / len(data_loader)
            scheduler.step(final_loss)

            if final_loss < best_loss:
                best_loss = final_loss
                patience = PATIENCE
                best_model = deepcopy(self.model).cpu()
            else:
                patience -= 1

            bar.set_description(f"loss={final_loss:.2f}")

            if patience == 0:
                break
        
        assert best_model != None
        self.model = best_model
                 

            
            
class SimpleCritic(Critic):
    def get_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.l1 = nn.Linear(1,1024)
                self.l2 = nn.Linear(1024,256)
                self.l3 = nn.Linear(256,1)

            def forward(self, x: pt.Tensor):
                x = pt.clip(x, *EXPLORATION_RANGE)
                x = self.l1(x)
                x = F.leaky_relu(x)
                x = self.l2(x)
                x = F.leaky_relu(x)
                x = self.l3(x)
                return x
        return Model()

    def plot(self, system: System):
        x = np.linspace(*EXPLORATION_RANGE, 50)

        self.model.eval()
        with pt.no_grad():
            y_critic = self.model(pt.tensor(x).float().unsqueeze(1).to(DEVICE)).cpu().numpy()
        
        y_real = np.array([system.get_solution([i]).score for i in x])

        plt.figure(figsize=(10, 6))
        
        plt.plot(x, y_real, label='Ground Truth (Real)', color='black', linestyle='--')
        plt.plot(x, y_critic, label='Critic Approximation', color='blue', alpha=0.8)
        plt.title('Comparison: Critic Model vs. Real System Solution')
        plt.xlabel('State Space ($x$)')
        plt.ylabel('Value ($V$)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.show()
        

        

class InertiaCritic(Critic):
    def get_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.l1 = nn.Linear(2,50)
                self.l2 = nn.Linear(50,50)
                self.l3 = nn.Linear(50,50)
                self.l4 = nn.Linear(50,1)

            def forward(self, x: pt.Tensor):
                x = self.l1(x)
                x = F.relu(x)
                x = self.l2(x)
                x = F.relu(x)
                x = self.l3(x)
                x = F.relu(x)
                x = self.l4(x)
                return x
        return Model()

    def plot(self, system: System):
        SPACE_POINTS = 5
        VELOCITY_POINTS = 5
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
                y_gt[i,j] = system.get_solution(state).score

        
        # Plot the surface
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(x, v, y_critic_numpy, cmap=cm.Blues)
        ax.plot_surface(x, v, y_gt, cmap=cm.Reds)

        ax.set(xticklabels=[],
               yticklabels=[],
               zticklabels=[])

        plt.show()


        # self.model.eval()
        # with pt.no_grad():
        #     y_critic = self.model(pt.tensor(x).float().unsqueeze(1).to(DEVICE)).cpu().numpy()


        # Plot the surface
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)

        
        # y_real = np.array([system.get_solution([i]).score for i in x])
        #
        # plt.figure(figsize=(10, 6))
        #
        # plt.plot(x, y_real, label='Ground Truth (Real)', color='black', linestyle='--')
        # plt.plot(x, y_critic, label='Critic Approximation', color='blue', alpha=0.8)
        # plt.title('Comparison: Critic Model vs. Real System Solution')
        # plt.xlabel('State Space ($x$)')
        # plt.ylabel('Value ($V$)')
        # plt.legend()
        # plt.grid(True, linestyle=':', alpha=0.6)
        #
        # plt.show()
        #
