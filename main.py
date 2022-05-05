import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from areas import Circle, Rectangle
from tasks import InsideTask, TotalBarrier
from factory_classes import LinearGamma


def calculate_control_input(x, t, total_barrier):
    # TODO

    def objective_function(u):
        return u[0]**2 + u[1]**2

    alpha = 1

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
    x0 = np.array([0, 0])

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
            plt.gca().add_artist(plt.Rectangle(area.bottom_left, area.width, area.height, fill=None))


def plot_path(states, areas, time_delta):
    plt.figure()
    plot_areas(areas)
    plt.axis('square')
    # TODO: don't hardcode these values
    plt.axis([-20, 20, -20, 20])

    '''for i in range(len(states)):
        states_per_cycle = 15
        if i % states_per_cycle == 0 or i+1 == len(states):
            plt.scatter(states[:i+1, 0], states[:i+1, 1], c=range(i+1), cmap=plt.cm.get_cmap('winter'))
            plt.pause(0.2 * states_per_cycle)'''

    plt.scatter(states[:, 0], states[:, 1], c=range(len(states)), cmap=plt.cm.get_cmap('winter'))


def plot_total_barrier(t_end, time_delta, saved_total_barrier):
    plt.figure()
    plt.plot(np.arange(0, t_end + time_delta, time_delta)[:len(saved_total_barrier)], saved_total_barrier)


def main():
    state = np.array([[1, 1]]).T
    saved_states = state.T
    t_end = 10

    areas = [
        Rectangle((0, 0), 40, 40),       # Work Area
        Circle((-17.5, 17.5), 2.5),  # Butter
        Rectangle((-12.5, 17.5), 5, 5),  # Sugar
        Rectangle((-7.5, 17.5), 5, 5),   # Eggs
        Rectangle((-2.5, 17.5), 5, 5),   # Flour
        Rectangle((2.5, 17.5), 5, 5),    # Baking Powder
        Rectangle((7.5, 17.5), 5, 5),    # Chocolate
        Rectangle((-17, 3), 6, 6),       # Water
        Rectangle((-17, 3), 6, 6),       # Milk
        Rectangle((-12, -16), 8, 8),     # Blenders
        Rectangle((16, 16), 8, 8),       # Ovens
        Rectangle((17, -17), 6, 6),      # Delivery Point
        Rectangle((0, -17), 6, 6),       # Starting/Ending Point
    ]

    # TODO: make sure eventually tasks are deactivated as soon as they are completed
    tasks = [
        #InsideTask(areas[0], gamma=LinearGamma(-20, 2, 'globally', (0, t_end))),
        InsideTask(areas[1], gamma=LinearGamma(-20, .1, 'eventually', (0, 5))),
        #InsideTask(areas[9], gamma=LinearGamma(-100, 1, 'eventually', (5, 10))),
        #InsideTask(areas[3], gamma=LinearGamma(-13, 2, 'globally', (25, 30))),
        #InsideTask(areas[9], gamma=LinearGamma(-14, 2, 'globally', (35, 40))),
        #InsideTask(areas[2], gamma=LinearGamma(-15, 2, 'globally', (45, 50))),
        #InsideTask(areas[9], gamma=LinearGamma(-16, 2, 'globally', (55, 60))),
        #InsideTask(areas[6], gamma=LinearGamma(-17, 2, 'globally', (65, 70))),
        #InsideTask(areas[9], gamma=LinearGamma(-18, 2, 'globally', (75, 80))),
        #InsideTask(areas[4], gamma=LinearGamma(-19, 2, 'globally', (85, 90))),
        #InsideTask(areas[9], gamma=LinearGamma(-20, 2, 'globally', (95, 100))),
        #InsideTask(areas[10], gamma=LinearGamma(-21, 2, 'globally', (105, 110))),
        #InsideTask(areas[11], gamma=LinearGamma(-22, 2, 'globally', (115, 120))),
        #InsideTask(areas[12], gamma=LinearGamma(-23, 2, 'eventually', (120, t_end))),
    ]

    total_barrier = TotalBarrier(tasks)

    # Define initial coordinates and goal area;
    # Define values for γ functions;
    # Define constants for collision avoidance;
    t = 0
    time_delta = 0.05
    saved_total_barrier = np.array([])
    while t < t_end:
        control_input = calculate_control_input(state, t, total_barrier)
        state = state + time_delta * control_input
        saved_states = np.append(saved_states, state.T, 0)
        saved_total_barrier = np.append(saved_total_barrier, total_barrier.evaluate(state, t))
        t += time_delta

    plot_total_barrier(t_end, time_delta, saved_total_barrier)
    plot_path(saved_states, areas, time_delta)
    plt.show()


if __name__ == '__main__':
    main()
