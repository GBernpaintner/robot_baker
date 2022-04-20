import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from areas import Circle, Rectangle
from tasks import InsideTask, TotalBarrier
from factory_classes import LinearGamma


def generate_total_barrier(tasks):
    def total_barrier(x, t):
        pass
    return total_barrier


def calculate_control_input(x, t, total_barrier):
    # TODO

    def objective_function(u):
        return u[0]**2 + u[1]**2

    alpha = 0.07

    b = alpha * total_barrier.evaluate(x, t) + total_barrier.evaluate_dbdt(x, t)

    # this should be greater than 0
    def constraint_function(u):
        u = np.resize(u, (2, 1))
        A = np.resize(total_barrier.evaluate_dbdx(x, t), (1, 2))
        return (A @ u + b)[0, 0]

    # type ineq means function should be non-negative
    constraints = {'type': 'ineq', 'fun': constraint_function}

    # u1 and u2 are unbounded
    bounds = [(None, None), (None, None)]

    # initial guess
    x0 = np.array([5, 5])

    # read docs for method
    result = minimize(objective_function, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    # new input in result.x
    u = np.resize(result.x, (2, 1))
    return u


def plot_areas(areas):
    for area in areas:
        if isinstance(area, Circle):
            plt.gca().add_artist(plt.Circle(area.center, area.radius, fill=None))
        elif isinstance(area, Rectangle):
            plt.gca().add_artist(plt.Rectangle(area.bottom_left, 40, 40, fill=None))


def plot_path(states, areas):
    plot_areas(areas)
    plt.scatter(states[:, 0], states[:, 1], c='#e377c2')
    plt.axis('square')
    # TODO: don't hardcode these values
    plt.axis([-20, 20, -20, 20])


def plot_total_barrier(t_end, time_delta, saved_total_barrier):
    plt.figure()
    print(saved_total_barrier)
    plt.plot(np.arange(0, t_end + time_delta, time_delta), saved_total_barrier)


def main():
    state = np.array([[1, 0]]).T
    saved_states = state.T
    t_end = 10

    areas = [
        Rectangle((0, 0), 40, 40),
        Circle((3, 1), 3),
        Circle((10, -7), 4)
    ]

    # TODO: add final-tag to ensure that last task stays active
    # TODO: make sure eventually tasks are deactivated as soon as they are completed
    # TODO: add eventually/globally tag
    tasks = [
        InsideTask(areas[0], gamma=LinearGamma(-20, 0.6, t_end)),
        InsideTask(areas[1], gamma=LinearGamma(-100, 0.06, 5))
    ]

    total_barrier = TotalBarrier(tasks)

    # Define initial coordinates and goal area;
    # Define values for γ functions;
    # Define constants for collision avoidance;
    t = 0
    time_delta = 0.2
    saved_total_barrier = np.array([])
    while t < t_end:
        control_input = calculate_control_input(state, t, total_barrier)
        state = state + time_delta * control_input
        saved_states = np.append(saved_states, state.T, 0)
        saved_total_barrier = np.append(saved_total_barrier, total_barrier.evaluate(state, t))
        t += time_delta

    plot_path(saved_states, areas)
    plot_total_barrier(t_end, time_delta, saved_total_barrier)
    plt.show()


if __name__ == '__main__':
    main()
