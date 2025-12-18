from abc import ABC, abstractmethod
import torch as pt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import RobotDataset
from parameters import DEVICE


class Critic(ABC):

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    def __init__(self, dataset: RobotDataset) -> None:
        self.dataset = dataset
        self.model = self.get_model()

    def train(self):
        
        data_loader = DataLoader(
            self.dataset,
            batch_size=64,
            shuffle=True
        )

        self.model.train()
        self.model.to(DEVICE)

        for _ in tqdm(range(100)):

            for s,t in data_loader:
                print(s)
                print(t)
            
            
class SimpleCritic(Critic):
    def get_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.l1 = nn.Linear(1,5)
                self.l2 = nn.Linear(5,10)
                self.l3 = nn.Linear(10,1)

            def forward(self, x):
                x = self.l1(x)
                x = F.relu(x)
                x = self.l2(x)
                x = F.relu(x)
                x = self.l2(x)
                x = F.relu(x)
                return x
        return Model()

