from typing import Final

# number of solutions that are generated for each training example
# the final example will be teh best solution among the N one.
NUM_SOLUTION_PER_DATAPOINT: Final[int] = 10

# the range we will collect samples in
EXPLORATION_RANGE: Final[tuple[float, float]] = (-2.15, 1.95)



