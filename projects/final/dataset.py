from abc import ABC, abstractmethod
from torch import dtype
from torch.utils.data import Dataset
import numpy as np
import os
from parameters import DATASET_SAVE_PATH, DATASET_SIZE, EXPLORATION_RANGE, VELOCITY_RANGE
import multiprocessing
import random
from system import InertiaSystem, SimpleSystem
from functools import reduce


DATASET_TYPE = tuple[np.typing.NDArray[np.float32], np.float32]

class RobotDataset(ABC, Dataset[DATASET_TYPE]):
    def __init__(self, cached=True):
        self.input: np.ndarray = np.array([]).reshape(0, 0)
        """
        shape [X,N] where X is input size and N is dataset size
        """
        self.output: np.ndarray = np.array([])
        """
        shape [N] where N is dataset size
        """

        loaded = False
        if cached:
            loaded = self.load()

        if not loaded:
            self.generate_dataset()
            self.save()

    @abstractmethod
    def get_name(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_example(num_elements: int) -> list[tuple[list[float], float]]:
        pass

    @property
    def output_file_name(self) -> str:
        return os.path.join(DATASET_SAVE_PATH, f"{self.get_name()}_output.csv")

    @property
    def input_file_name(self) -> str:
        return os.path.join(DATASET_SAVE_PATH, f"{self.get_name()}_input.csv")

    def save(self):
        os.makedirs(DATASET_SAVE_PATH, exist_ok=True)
        input_np = self.input.T 
        output_np = self.output

        np.savetxt(self.input_file_name, input_np, delimiter=",")
        np.savetxt(self.output_file_name, output_np, delimiter=",")

    def load(self) -> bool:
        if not (os.path.exists(self.input_file_name) and os.path.exists(self.output_file_name)):
            return False
        try:
            input_np = np.loadtxt(self.input_file_name, delimiter=",")
            output_np = np.loadtxt(self.output_file_name, delimiter=",")

            self.input = input_np.T.astype(np.float32)
            self.output = output_np.astype(np.float32)
            
            return True
        except Exception as _:
            return False

    def generate_dataset(self):
        pool = multiprocessing.Pool()

        examples = reduce(
            lambda a,b: a+b,
            pool.map(
                self.__class__.get_example,
                [100 for _ in range(DATASET_SIZE//100)] + [DATASET_SIZE % 100]
            )
        )

        input = [i for i,_ in examples]
        output = [o for _,o in examples]

        self.input = np.asarray(input, dtype=np.float32).T
        self.output = np.asarray(output, dtype=np.float32)



    def __len__(self):
        return self.output.shape[0]

    def __getitem__(self, idx: int) -> DATASET_TYPE:
        # return np.asarray([float(idx)],dtype=np.float32), np.float32(idx)
        if len(self.input.shape) == 1:
            return self.input[idx], self.output[idx]
        return self.input[:, idx], self.output[idx]



class SimpleRobotDataset(RobotDataset):
    def get_name(self) -> str:
        return "simple"

    @staticmethod
    def get_example(num_elements: int) -> list[tuple[list[float], float]]:
        s = InertiaSystem()
        to_return: list[tuple[list[float], float]] = []
        for _ in range(num_elements):
            s = SimpleSystem()
            point = random.uniform(*EXPLORATION_RANGE)
            score = s.get_solution([point]).score
            to_return.append(([point], score))
        return to_return


class InertiaRobotDataset(RobotDataset):
    def get_name(self) -> str:
        return "inertia"

    @staticmethod
    def get_example(num_elements: int) -> list[tuple[list[float], float]]:
        s = InertiaSystem()
        to_return: list[tuple[list[float], float]] = []
        for _ in range(num_elements):
            point = random.uniform(*EXPLORATION_RANGE)
            velocity = random.uniform(*VELOCITY_RANGE)
            state = [point, velocity]
            score = s.get_solution(state).score
            to_return.append((state, score))
        return to_return
