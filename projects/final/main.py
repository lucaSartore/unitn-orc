from dataset import InertiaRobotDataset, SimpleRobotDataset
from critic import InertiaCritic, SimpleCritic
from datetime import datetime
def main():

    # d = SimpleRobotDataset()
    # c = SimpleCritic(d)
    # c.run()

    d = InertiaRobotDataset()
    c = InertiaCritic(d)
    c.run()


if __name__ == '__main__':
    main()

