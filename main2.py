import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import log, exp

from areas import Circle, Rectangle


class InsideCirclePredicateFunction:
    def __init__(self, circle):
        self.circle = circle

    def __call__(self, x):
        # v = center - x
        v = np.array([self.circle.center]).T - x
        return self.circle.radius**2 - v[0, 0]**2 - v[1, 0]**2

    def gradient(self, x):
        gradient_x = 2 * (self.circle.center[0] - x[0, 0])
        gradient_y = 2 * (self.circle.center[1] - x[1, 0])
        return np.array([[gradient_x, gradient_y]]).T


'''h = InsideCirclePredicateFunction(Circle((1, 1), 1))
x = np.array([[0, 0]]).T
print(h(x))
print(h.gradient(x))'''


class InsideRectanglePredicateFunction:
    def __init__(self, rectangle, side):
        self.rectangle = rectangle
        if side not in ('top', 'left', 'bottom', 'right'):
            raise ValueError('Variable side must be "top", "left", "bottom" or "right".')
        self.side = side

    def __call__(self, x):
        if self.side == 'top':
            top = self.rectangle.center[1] + self.rectangle.height / 2
            return top - x[1, 0]
        elif self.side == 'left':
            left = self.rectangle.center[0] - self.rectangle.width / 2
            return x[0, 0] - left
        elif self.side == 'bottom':
            bottom = self.rectangle.center[1] - self.rectangle.height / 2
            return x[1, 0] - bottom
        elif self.side == 'right':
            right = self.rectangle.center[0] + self.rectangle.width / 2
            return right - x[0, 0]

    def gradient(self, x):
        if self.side == 'top':
            return np.array([[0, -1]]).T
        elif self.side == 'left':
            return np.array([[1, 0]]).T
        elif self.side == 'bottom':
            return np.array([[0, 1]]).T
        elif self.side == 'right':
            return np.array([[-1, 0]]).T


'''rect = Rectangle((0, 0), 1, 1)
x = np.array([[-1], [-1]])
h2a = InsideRectanglePredicateFunction(rect, 'top')
h2b = InsideRectanglePredicateFunction(rect, 'left')
h2c = InsideRectanglePredicateFunction(rect, 'bottom')
h2d = InsideRectanglePredicateFunction(rect, 'right')
print('top   ', h2a(x))
print('left  ', h2b(x))
print('bottom', h2c(x))
print('right ', h2d(x))
print('top.grad   ', h2a.gradient(x).T)
print('left.grad  ', h2b.gradient(x).T)
print('bottom.grad', h2c.gradient(x).T)
print('right.grad ', h2d.gradient(x).T)'''


class GammaFunction:
    def __init__(self, type, range, gamma_0, gamma_inf):
        if type not in ('globally', 'eventually'):
            raise ValueError('Variable side must be "eventually" or "globally".')
        self.type = type
        self.a, self.b = range
        if self.type == 'globally':
            self.t_star = self.a
        elif self.type == 'eventually':
            self.t_star = self.b
        self.gamma_0 = gamma_0
        self.gamma_inf = gamma_inf

    def __call__(self, t):
        if t >= self.t_star:
            return self.gamma_inf
        else:
            return (self.gamma_inf - self.gamma_0) * t / self.t_star + self.gamma_0

    def gradient(self, t):
        if t >= self.t_star:
            return 0
        else:
            return (self.gamma_inf - self.gamma_0) / self.t_star


'''# noinspection NonAsciiCharacters
γ1 = GammaFunction('globally', (5, 10), -20, 0.1)
# noinspection NonAsciiCharacters
γ2 = GammaFunction('eventually', (5, 10), -20, 0.1)
for t in range(16):
    print(f'γ1({t}) =', γ1(t), f'γ1.gradient({t}) =', γ1.gradient(t))
for t in range(16):
    print(f'γ2({t}) =', γ2(t), f'γ2.gradient({t}) =', γ2.gradient(t))'''


class CandidateControlBarrierFunction:
    def __init__(self, predicate, gamma, always_active=False):
        self.predicate = predicate
        self.gamma = gamma
        self.always_active = always_active
        self._deactivated = False
        self.connected_candidate_cbfs = None

    def is_active(self, x, t):
        if self.always_active:
            return True
        if self._deactivated:
            return False
        if t > self.gamma.b:
            self._deactivated = True
            return False
        if self.gamma.type == 'eventually' and t >= self.gamma.a:
            if self.connected_candidate_cbfs:
                connected_should_deactivate = []
                for candidate_cbf in self.connected_candidate_cbfs:
                    connected_should_deactivate.append(candidate_cbf.predicate(x) >= candidate_cbf.gamma.gamma_inf)
                if all(connected_should_deactivate):
                    for candidate_cbf in self.connected_candidate_cbfs:
                        candidate_cbf._deactivated = True
                    return False
            elif self.predicate(x) >= self.gamma.gamma_inf:
                self._deactivated = True
                return False
        return True

    def reset(self):
        self._deactivated = False

    def __call__(self, x, t):
        return -self.gamma(t) + self.predicate(x)

    def gradient_x(self, x, t):
        return self.predicate.gradient(x)

    def gradient_t(self, x, t):
        return -self.gamma.gradient(t)

    def __and__(self, other):
        if isinstance(other, ControlBarrierFunction):
            CBF = other.clone()
            CBF.append(self)
            return CBF
        elif isinstance(other, CandidateControlBarrierFunction):
            CBF = ControlBarrierFunction()
            CBF.append(self)
            CBF.append(other)
            return CBF


def connect_candidate_cbfs(*candidate_cbfs):
    # TODO control same gamma.a, gamma.b
    for candidate_cbf in candidate_cbfs:
        candidate_cbf.connected_candidate_cbfs = candidate_cbfs


class ControlBarrierFunction:
    def __init__(self):
        self.candidate_cbfs = []

    def append(self, candidate_cbf):
        self.candidate_cbfs.append(candidate_cbf)

    def reset(self):
        for candidate_cbf in self.candidate_cbfs:
            candidate_cbf.reset()

    def __call__(self, x, t):
        terms = []
        for candidate_cbf in self.candidate_cbfs:
            if candidate_cbf.is_active(x, t):
                terms.append(exp(-candidate_cbf(x, t)))
        return -log(sum(terms))

    def gradient_x(self, x, t):
        numerator_terms = []
        denominator_terms = []
        for candidate_cbf in self.candidate_cbfs:
            if candidate_cbf.is_active(x, t):
                numerator_terms.append(exp(-candidate_cbf(x, t)) * candidate_cbf.gradient_x(x, t))
                denominator_terms.append(exp(-candidate_cbf(x, t)))
        return sum(numerator_terms) / sum(denominator_terms)

    def gradient_t(self, x, t):
        numerator_terms = []
        denominator_terms = []
        for candidate_cbf in self.candidate_cbfs:
            if candidate_cbf.is_active(x, t):
                numerator_terms.append(exp(-candidate_cbf(x, t)) * candidate_cbf.gradient_t(x, t))
                denominator_terms.append(exp(-candidate_cbf(x, t)))
        return sum(numerator_terms) / sum(denominator_terms)

    def clone(self):
        clone = ControlBarrierFunction()
        # TODO clone candidate CBFs as well?
        clone.candidate_cbfs = [*self.candidate_cbfs]
        return clone

    def __and__(self, other):
        if isinstance(other, CandidateControlBarrierFunction):
            CBF = self.clone()
            CBF.append(other)
            return CBF
        elif isinstance(other, ControlBarrierFunction):
            CBF = ControlBarrierFunction()
            CBF.candidate_cbfs = [*self.candidate_cbfs, *other.candidate_cbfs]
            return CBF


def calculate_control_input(x, t, alpha, cbf):

    def objective_function(u):
        return u[0]**2 + u[1]**2

    b = alpha * cbf(x, t) + cbf.gradient_t(x, t)

    # this should be greater than 0
    def constraint_function(u):
        u = np.resize(u, (2, 1))
        A = np.resize(cbf.gradient_x(x, t), (1, 2))
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


def plot_path(states, areas):
    plt.figure()
    plot_areas(areas)
    plt.axis('square')
    # TODO: don't hardcode these values
    plt.axis([-2, 2, -2, 2])

    '''for i in range(len(states)):
        states_per_cycle = 15
        if i % states_per_cycle == 0 or i+1 == len(states):
            plt.scatter(states[:i+1, 0], states[:i+1, 1], c=range(i+1), cmap=plt.cm.get_cmap('winter'))
            plt.pause(0.2 * states_per_cycle)'''

    plt.scatter(states[:, 0], states[:, 1], c=range(len(states)), cmap=plt.cm.get_cmap('winter'))


def plot_cbf(t_end, time_delta, saved_cbfs):
    plt.figure()
    plt.plot(np.arange(0, t_end + time_delta, time_delta)[:len(saved_cbfs)], saved_cbfs)


def main():
    state = np.array([[0, 0]]).T
    saved_states = state.T
    t_end = 20

    '''rectangle = Rectangle((2, 2), 2, 2)
    circle = Circle((0, 0), 1)
    areas = [rectangle, circle]

    rect_gamma = GammaFunction('eventually', (5, 10), -20, 0.1)
    rect_top = CandidateControlBarrierFunction(InsideRectanglePredicateFunction(rectangle, 'top'), rect_gamma)
    rect_left = CandidateControlBarrierFunction(InsideRectanglePredicateFunction(rectangle, 'left'), rect_gamma)
    rect_bottom = CandidateControlBarrierFunction(InsideRectanglePredicateFunction(rectangle, 'bottom'), rect_gamma)
    rect_right = CandidateControlBarrierFunction(InsideRectanglePredicateFunction(rectangle, 'right'), rect_gamma)
    connect_candidate_cbfs(rect_top, rect_left, rect_bottom, rect_right)

    circ_gamma = GammaFunction('globally', (12, 15), -40, 0.1)
    circ = CandidateControlBarrierFunction(InsideCirclePredicateFunction(circle), circ_gamma, always_active=True)

    cbf = rect_top & rect_left & rect_bottom & rect_right & circ'''

    '''state = np.array([[0, 0]]).T
    t = 10

    print(f'        CBF({state.T}.T, {t}) =', CBF(state, t))
    print(f'   rect_top({state.T}.T, {t}) =', rect_top(state, t), f'active: {rect_top.is_active(state, t)}')
    print(f'  rect_left({state.T}.T, {t}) =', rect_left(state, t), f'active: {rect_left.is_active(state, t)}')
    print(f'rect_bottom({state.T}.T, {t}) =', rect_bottom(state, t), f'active: {rect_bottom.is_active(state, t)}')
    print(f' rect_right({state.T}.T, {t}) =', rect_right(state, t), f'active: {rect_right.is_active(state, t)}')
    print(f'       circ({state.T}.T, {t}) =', circ(state, t), f'active: {circ.is_active(state, t)}')'''

    areas = [
        Rectangle((0, 0), 4, 4),   #  0 - Work Area
        Circle((-1.75, 1.75), .25),  #  1 - Butter
        # Circle((-1.25, 1.75), .25),  #  2 - Sugar
        # Circle((-.75, 1.75), .25),   #  3 - Eggs
        # Circle((-.25, 1.75), .25),   #  4 - Flour
        # Circle((.25, 1.75), .25),    #  5 - Baking Powder
        # Circle((.75, 1.75), .25),    #  6 - Chocolate
        # Circle((-1.7, .3), .3),      #  7 - Water
        # Circle((-1.7, .3), .3),      #  8 - Milk
        Circle((-1.2, -1.6), .4),    #  9 - Blenders
        # Circle((1.6, 1.6), .4),      # 10 - Ovens
        # Circle((1.7, -1.7), .3),     # 11 - Delivery Point
        # Circle((0, -1.7), .3),       # 12 - Starting/Ending Point
    ]

    alpha = 0.07
    time_delta = 0.01

    bounds_gamma = GammaFunction('eventually', (0, t_end), 0, 0.01)
    bounds_top_predicate = InsideRectanglePredicateFunction(areas[0], 'top')
    bounds_left_predicate = InsideRectanglePredicateFunction(areas[0], 'left')
    bounds_bottom_predicate = InsideRectanglePredicateFunction(areas[0], 'bottom')
    bounds_right_predicate = InsideRectanglePredicateFunction(areas[0], 'right')
    bounds_top = CandidateControlBarrierFunction(bounds_top_predicate, bounds_gamma, always_active=True)
    bounds_left = CandidateControlBarrierFunction(bounds_left_predicate, bounds_gamma, always_active=True)
    bounds_bottom = CandidateControlBarrierFunction(bounds_bottom_predicate, bounds_gamma, always_active=True)
    bounds_right = CandidateControlBarrierFunction(bounds_right_predicate, bounds_gamma, always_active=True)

    connect_candidate_cbfs(bounds_top, bounds_left, bounds_bottom, bounds_right)

    bounds_candidate = bounds_top & bounds_left & bounds_bottom & bounds_right

    gamma0 = GammaFunction('eventually', (5, 8), -100, 0.01)
    predicate0 = InsideCirclePredicateFunction(areas[1])
    candidate0 = CandidateControlBarrierFunction(predicate0, gamma0)

    gamma1 = GammaFunction('eventually', (13, 16), -200, 0.01)
    predicate1 = InsideCirclePredicateFunction(areas[2])
    candidate1 = CandidateControlBarrierFunction(predicate1, gamma1)

    # gamma2 = GammaFunction('eventually', (16, 20), -400, 0.1)
    # predicate2 = InsideCirclePredicateFunction(areas[11])
    # candidate2 = CandidateControlBarrierFunction(predicate2, gamma2, always_active=True)

    cbf = bounds_candidate & candidate0 #& candidate1  # & candidate2

    '''# TODO: make sure eventually tasks are deactivated as soon as they are completed
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

    total_barrier = TotalBarrier(tasks)'''

    # Define initial coordinates and goal area;
    # Define values for γ functions;
    # Define constants for collision avoidance;
    t = 0
    saved_cbfs = np.array([])
    while t < t_end:
        control_input = calculate_control_input(state, t, alpha, cbf)
        print(control_input, t)
        state = state + time_delta * control_input
        saved_states = np.append(saved_states, state.T, 0)
        saved_cbfs = np.append(saved_cbfs, cbf(state, t))
        t += time_delta

    plot_cbf(t_end, time_delta, saved_cbfs)

    plot_path(saved_states, areas)
    plt.show()


if __name__ == '__main__':
    main()
