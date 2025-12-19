from dataset import InertiaRobotDataset, SimpleRobotDataset
from critic import SimpleCritic


def main():
    d = SimpleRobotDataset()
    c = SimpleCritic(d)
    c.run()


if __name__ == '__main__':
    main()

