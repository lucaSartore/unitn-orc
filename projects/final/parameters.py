from typing import Final

# number of solutions that are generated for each training example
# the final example will be teh best solution among the N one.
NUM_SOLUTION_PER_DATAPOINT: Final[int] = 10

# the range we will collect samples in
EXPLORATION_RANGE: Final[tuple[float, float]] = (-2.15, 1.95)

# the range of the initial velocity of the robot
VELOCITY_RANGE: Final[tuple[float, float]] = (-5, 5)

# device used in pytorch
DEVICE: Final[str] = 'cuda'

# path where the dataset is saved
DATASET_SAVE_PATH: Final[str] = './dataset/'

# number of training samples in the dataset
DATASET_SIZE: Final[int] = 1_000

# number of steps without a loss reduction that are accepted
# by the optimization algorithm
PATIENCE: Final[int] = 50

# maximum number of generation if patience doesn't terminate
# the training earlier
MAX_TRAINING_GENERATIONS: Final[int] = 10_000

# train test dataset split used for training both the critic
# and the actor
TRAIN_TEST_SPLIT: Final[list[float]] = [0.9, 0.1]


