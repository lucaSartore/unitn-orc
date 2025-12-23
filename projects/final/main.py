from dataclasses import dataclass
from typing import Callable, Literal
from actor import Actor, InertiaActor, SimpleActor
from dataset import InertiaRobotDataset, RobotDataset, SimpleRobotDataset
from critic import Critic, InertiaCritic, SimpleCritic
from system import InertiaSystem, SimpleSystem, System
from time import time

INTERACTIVE: bool = True # if true plots will be shown, otherwise they will be saved
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
            [0.7, 2],
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

    name: str
    f_name = lambda name: None if INTERACTIVE else name
    
    # initializing the system
    s = config.system() #type: ignore
    d = config.dataset()
    c = config.critic(d)

    # training the critic
    c.run()

    if PLOT_CRITIC_FUNCTION:
        print("Plotting critic function")
        c.plot(s, f_name(f"{TEST_TYPE}_system___critic_function.png"))

    # creating the actor 
    a = config.actor(s,c)
    # training the actor
    a.run()

    if PLOT_ACTOR_FUNCTION:
        print("Plotting actor function")
        a.plot(s, f_name(f"{TEST_TYPE}_system___actor_function.png"))


    print("Plotting actor vs ground trough trajectories")
    policy = a.get_policy()
    for i, state in enumerate(config.states_to_test):
        s1 = s.evaluate_policy(policy, state)
        s2 = s.get_solution(state)
        s.plot_multiple_solutions(
            [s1, s2],
            labels= [
                "actor",
                "optimal control"
            ],
            file_name = f_name(f"./images/{TEST_TYPE}___system__actor_vs_ground_trough___{i}.png")
        )


    print("Testing execution time for various OCP")
    def fn[T](to_test: Callable[[],T]) -> tuple[T,float]:
        start = time()
        result = to_test()
        end = time()
        return result, end-start
    for i, state in enumerate(config.states_to_test):
        s1, t1 = fn(lambda: s.get_solution(state, 1))
        s2, t2 = fn(lambda: s.get_solution(state, 10))
        s3, t3 = fn(lambda: s.evaluate_policy(a.get_policy(), state))
        s4, t4 = fn(lambda: s.get_solution(
            state,
            1,
            s.get_initial_guess_from_policy(a.get_policy(), state)
        ))

        s.plot_multiple_solutions(
            [s1, s2, s3, s4],
            labels= [
                f"Optimal Control (best of one) t={t1:.2f} [s]",
                f"Optimal Control (best of ten) t={t2:.2f} [s]",
                f"Actor t={t3:.2f} [s]",
                f"Optimal Control + Actor initialization t={t4:.2f} [s]",
            ],
            file_name = f_name(f"./images/{TEST_TYPE}_system___execution_time_test___{i}.png")
        )

if __name__ == '__main__':
    main()
