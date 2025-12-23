from actor import SimpleActor
from dataset import InertiaRobotDataset, SimpleRobotDataset
from critic import InertiaCritic, SimpleCritic
from datetime import datetime

from parameters import DEVICE
from system import InertiaSystem, SimpleSystem
def main():


    s = InertiaSystem()
    d = InertiaRobotDataset()
    c = InertiaCritic(d)
    c.run()
    c.plot(s)
    return

    s = SimpleSystem()
    d = SimpleRobotDataset()
    c = SimpleCritic(d)
    
    c.run()

    # c.plot(s)
    # return

    # import torch
    # for d in [-1.9, -1.8, -1.7]:
    #     print(f"evaluating accuracy in pont {d}")
    #     print(f"critic score: ", c.model(torch.tensor([d], dtype=torch.float32).to(DEVICE)).item())
    #     print(f"system score: ", s.get_solution([d]).score)
    # return

    a = SimpleActor(s,c)
    a.run()

    # a.plot(s)

    policy = a.get_policy()
    for d in [-1.9, 1.0, 2.0, 0.5, -1.2]:
        s1 = s.evaluate_policy(policy, [d])
        s2 = s.get_solution([d])
        s.plot_multiple_solutions(
            [s1, s2],
            labels= [
                "actor",
                "optimal control"
            ]
        )
    return


    
    # d = InertiaRobotDataset()
    # c = InertiaCritic(d)
    # c.run()
    

if __name__ == '__main__':
    main()

