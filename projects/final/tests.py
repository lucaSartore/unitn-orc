from parameters import EXPLORATION_RANGE
from system import GreedyPolicy, SimpleSystem, Solution, InertiaSystem
from datetime import datetime

def main():
    # test_all_systems()
    performance_test()

def performance_test():
    N = 40
    s =  InertiaSystem()
    t = datetime.now()
    for _ in range(N):
        s.get_solution([-1.1, 0])
    dt = datetime.now() - t
    print(f"time to generate {N} solutions is: {dt}")

def test_all_systems():
    solutions: list[Solution] = []
    s = SimpleSystem()
    solutions.append(s.get_solution([-1.1]))
    s.plot_last_solution()
    solutions.append(s.evaluate_policy(GreedyPolicy(), [-1.1]))
    s.plot_last_solution()

    s =  InertiaSystem()
    solutions.append(s.get_solution([-1.1, 0]))
    s.plot_last_solution()
    solutions.append(s.evaluate_policy(GreedyPolicy(), [-1.1,0]))
    s.plot_last_solution()

    labels = ['simple OCP', 'simple greedy', 'inertia OCP', 'inertia greedy']

    for sol, label in zip(solutions, labels):
        print(f"solution {label} has score {sol.score}")

    s.plot_multiple_solutions(
        solutions,
        labels= labels,
        y_range= EXPLORATION_RANGE
    )



if __name__ == '__main__':
    main()
