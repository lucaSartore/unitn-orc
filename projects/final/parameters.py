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
DATASET_SIZE: Final[int] = 10_000

# number of steps without a loss reduction that are accepted
# by the optimization algorithm
PATIENCE: Final[int] = 50

# train test dataset split used for training both the critic
# and the actor
TRAIN_TEST_SPLIT: Final[list[float]] = [0.9, 0.1]

# number of cores that are used for dataset generation
# if none the system's number of cores will be used
# (note that using none may result in excessive memory usage)
CORES_FOR_DATASET_GENERATION: Final[int | None] = 6

# number of examples that each core should generation.
# having a large number here allow the cost of spawning
# a process to be amortized
GENERATED_POINTS_PER_CORE: Final[int] = 50

# maximum number of generation if patience doesn't terminate
# the training earlier
CRITIC_MAX_TRAINING_GENERATIONS: Final[int] = 2000

# learning rate used by the optimizer for the
# critic
CRITIC_LEARNING_RATE: Final[float] = 0.0001

# batch size using in the critic training process
CRITIC_BATCH_SIZE: Final[int] = 64

# maximum number of generation if patience doesn't terminate
# the training earlier
ACTOR_MAX_TRAINING_GENERATIONS: Final[int] = 5000

# learning rate used by the optimizer for the
# actor
ACTOR_LEARNING_RATE: Final[float] = 0.0005

# batch size using in the actor training process
ACTOR_BATCH_SIZE: Final[int] = 2048
