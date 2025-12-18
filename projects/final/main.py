from dataset import InertiaRobotDataset, SimpleRobotDataset
from critic import SimpleCritic


def main():
    d = SimpleRobotDataset()
    c = SimpleCritic(d)
    c.train()


if __name__ == '__main__':
    main()

