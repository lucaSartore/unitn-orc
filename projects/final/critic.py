from abc import ABC, abstractmethod
import math
import torch as pt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataset import RobotDataset
from parameters import DEVICE, MAX_TRAINING_GENERATIONS, PATIENCE, TRAIN_TEST_SPLIT
from copy import deepcopy


class Critic(ABC):

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    def __init__(self, dataset: RobotDataset) -> None:
        self.dataset = dataset
        self.model = self.get_model()

    def run(self):
        train_set, val_set = pt.utils.data.random_split(self.dataset, TRAIN_TEST_SPLIT)
        self.train(train_set)
        self.validate(val_set)
        pass

    def validate(self, dataset: Dataset):
        data_loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True
        )

        self.model.eval()
        self.model = self.model.to(DEVICE)

        running_loss = 0
        input: pt.Tensor
        output: pt.Tensor
        for input,output in data_loader:
            input = input.to(DEVICE)
            input = input.unsqueeze(1)
            output = output.to(DEVICE)
            output = output.unsqueeze(1)

            predicted_output: pt.Tensor = self.model(input)
            loss = F.mse_loss(predicted_output, output)
            # loss = F.l1_loss(predicted_output, output)
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

        # optimizer = pt.optim.SGD(params=self.model.parameters(), lr=0.001)
        optimizer = pt.optim.AdamW(params=self.model.parameters())

        
        best_model: nn.Module | None = None
        patience = PATIENCE
        best_loss = math.inf
        bar = tqdm(range(MAX_TRAINING_GENERATIONS))
        for _ in bar:

            running_loss = 0
            input: pt.Tensor
            output: pt.Tensor
            for input,output in data_loader:
                input = input.to(DEVICE)
                input = input.unsqueeze(1)
                output = output.to(DEVICE)
                output = output.unsqueeze(1)
                optimizer.zero_grad()

                predicted_output: pt.Tensor = self.model(input)
                loss = F.mse_loss(predicted_output, output)
                # loss = F.l1_loss(predicted_output, output)
                # loss = F.huber_loss(predicted_output, output)
                loss.backward()

                print(input[0], output[0], predicted_output[0])
                optimizer.step()
                
                running_loss += loss.item()

            final_loss = running_loss / len(data_loader)

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

class InertiaCritic(Critic):
    def get_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.l1 = nn.Linear(2,30)
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


