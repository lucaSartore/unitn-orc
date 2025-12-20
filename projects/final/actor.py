from abc import ABC, abstractmethod
import math
import torch as pt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from critic import Critic
from dataset import RobotDataset
from parameters import ACTOR_MAX_TRAINING_GENERATIONS, CRITIC_BATCH_SIZE, DEVICE, CRITIC_LEARNING_RATE, CRITIC_MAX_TRAINING_GENERATIONS, EXPLORATION_RANGE, PATIENCE, TRAIN_TEST_SPLIT
from copy import deepcopy
from system import System



class Actor(ABC):

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    @abstractmethod
    def get_random_state(self, batch_size: int) -> pt.Tensor:
        pass

    def __init__(self, system: System, critic: Critic) -> None:
        self.system = system
        self.critic = critic.model
        self.model = self.get_model()

    def run(self):
        self.train()
        self.validate()

    def validate(self):
        pass

    def train(self):
        
        self.critic = self.critic.eval()
        self.model = self.model.train()


        bar = tqdm(range(ACTOR_MAX_TRAINING_GENERATIONS))
        for _ in bar:
            input = self.get_random_state(CRITIC_BATCH_SIZE)
            action: pt.Tensor = self.model(input)

            state_cost = self.system.cost_function(input, action)
            next_state = self.system.state_transition_function(input, action)[0]
            


class SimpleActor(Actor):
    def get_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.l1 = nn.Linear(1,30)
                self.l2 = nn.Linear(30,30)
                self.l3 = nn.Linear(30,30)
                self.l4 = nn.Linear(30,1)

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

    def get_random_state(self, batch_size: int) -> pt.Tensor:
        start, end = EXPLORATION_RANGE
        x = pt.rand((batch_size,1)) * (end-start) + start
        return x.to(DEVICE)

