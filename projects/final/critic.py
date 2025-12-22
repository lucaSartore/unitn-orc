from abc import ABC, abstractmethod
import math
import torch as pt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataset import RobotDataset
from parameters import CRITIC_BATCH_SIZE, DEVICE, CRITIC_LEARNING_RATE, CRITIC_MAX_TRAINING_GENERATIONS, EXPLORATION_RANGE, PATIENCE, TRAIN_TEST_SPLIT
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

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
            # loss = F.huber_loss(predicted_output, output)

            running_loss += loss.item()

        final_loss = running_loss / len(data_loader)

        print(f"Final loss with validation dataset: {final_loss}")
                 

    def train(self, dataset: Dataset):
        
        data_loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True
        )

        self.model.train()
        self.model = self.model.to(DEVICE)

        optimizer = pt.optim.AdamW(params=self.model.parameters(), lr=0.0001)

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
                # loss = F.huber_loss(predicted_output, output)
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
                self.l1 = nn.Linear(1,5000)
                self.l2 = nn.Linear(5000,256)
                self.l3 = nn.Linear(256,1)
                # self.l4 = nn.Linear(128,64)
                # self.l5 = nn.Linear(64,1)

            def forward(self, x: pt.Tensor):
                x = pt.clip(x, *EXPLORATION_RANGE)
                x = self.l1(x)
                x = F.leaky_relu(x)
                x = self.l2(x)
                # x = F.leaky_relu(x)
                # x = self.l3(x)
                # x = F.leaky_relu(x)
                # x = self.l4(x)
                x = F.leaky_relu(x)
                x = self.l3(x)
                return x
        return Model()

    def plot(self, system: System):
        # x = np.linspace(*EXPLORATION_RANGE, 50)
        x = np.linspace(-1.9, -1.7, 50)
        # Ensure model is in eval mode and move tensor to CPU for plotting
        self.model.eval()
        with pt.no_grad():
            y_critic = self.model(pt.tensor(x).float().unsqueeze(1).to(DEVICE)).cpu().numpy()
        
        # Convert the real solution list to a numpy array
        y_real = np.array([system.get_solution([i]).score for i in x])

        plt.figure(figsize=(10, 6))
        
        # Plotting the data
        plt.plot(x, y_real, label='Ground Truth (Real)', color='black', linestyle='--')
        plt.plot(x, y_critic, label='Critic Approximation', color='blue', alpha=0.8)
        
        # Formatting the chart
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
