from dataclasses import dataclass
from typing import Literal
from actor import Actor, InertiaActor, SimpleActor
from dataset import InertiaRobotDataset, RobotDataset, SimpleRobotDataset
from critic import Critic, InertiaCritic, SimpleCritic
from system import InertiaSystem, SimpleSystem, System


TEST_TYPE: Literal['simple', 'inertia'] = 'inertia'
PLOT_CRITIC_FUNCTION: bool = False
PLOT_ACTOR_FUNCTION: bool = False


@dataclass
class TestConfig:
    system: type[System]
    dataset: type[RobotDataset]
    critic: type[Critic]
    actor: type[Actor]
    states_to_test: list[list[float]]

    
def main():

    simple_config = TestConfig(
        system= SimpleSystem,
        dataset= SimpleRobotDataset,
        critic= SimpleCritic,
        actor = SimpleActor,
        states_to_test=[
            [-1.9],
            [-1.2],
            [0.5],
            [1.0],
            [2.0]
        ]
    )

    inertia_config = TestConfig(
        system= InertiaSystem,
        dataset= InertiaRobotDataset,
        critic= InertiaCritic,
        actor = InertiaActor,
        states_to_test=[
            [-1.9,-1],
            [-1.2,0],
            [0.5, 2],
            [0.5, 0],
            [1.0, 3],
            [2.0,0],
            [2.0,-3]
        ]
    )
    
    if TEST_TYPE == 'simple':
        run_test(simple_config)
    else:
        run_test(inertia_config)

def run_test(config: TestConfig):

    
    # initializing the system
    s = config.system() #type: ignore
    d = config.dataset()
    c = config.critic(d)

    # training the critic
    c.run()

    if PLOT_CRITIC_FUNCTION:
        print("Plotting critic function")
        c.plot(s)

    # creating the actor 
    a = config.actor(s,c)
    # training the actor
    a.run()

    if PLOT_ACTOR_FUNCTION:
        print("Plotting actor function")
        a.plot(s)


    print("Plotting actor vs ground trough trajectories")
    policy = a.get_policy()
    for state in config.states_to_test:
        s1 = s.evaluate_policy(policy, state)
        s2 = s.get_solution(state)
        s.plot_multiple_solutions(
            [s1, s2],
            labels= [
                "actor",
                "optimal control"
            ]
        )


    
if __name__ == '__main__':
    main()

