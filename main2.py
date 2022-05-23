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
        return self.circle.radius - np.linalg.norm(v)

    def gradient(self, x):
        gradient_x = -(x[0, 0] - self.circle.center[0]) / (np.linalg.norm(np.array([self.circle.center]).T - x) + 0.00001)
        gradient_y = -(x[1, 0] - self.circle.center[1]) / (np.linalg.norm(np.array([self.circle.center]).T - x) + 0.00001)
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
    """Ensures that all the connected candidate CBFs are activated/deactivated at the same time."""
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


def create_candidate(gamma_type, gamma_range, gamma_0, gamma_inf, area, always_active=False):
    gamma = GammaFunction(gamma_type, gamma_range, gamma_0, gamma_inf)
    if isinstance(area, Rectangle):
        top_predicate = InsideRectanglePredicateFunction(area, 'top')
        left_predicate = InsideRectanglePredicateFunction(area, 'left')
        bottom_predicate = InsideRectanglePredicateFunction(area, 'bottom')
        right_predicate = InsideRectanglePredicateFunction(area, 'right')
        top = CandidateControlBarrierFunction(top_predicate, gamma, always_active)
        left = CandidateControlBarrierFunction(left_predicate, gamma, always_active)
        bottom = CandidateControlBarrierFunction(bottom_predicate, gamma, always_active)
        right = CandidateControlBarrierFunction(right_predicate, gamma, always_active)
        connect_candidate_cbfs(top, left, bottom, right)
        return top & left & bottom & right
    elif isinstance(area, Circle):
        predicate = InsideCirclePredicateFunction(area)
        return CandidateControlBarrierFunction(predicate, gamma, always_active)


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
    bounds = [(-15, 15), (-15, 15)]

    # initial guess
    x0 = np.array([0, 0])

    # read docs for method
    result = minimize(objective_function, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    # new input in result.x
    u = np.resize(result.x, (2, 1))
    return u


def plot_areas(areas):
    i = 1
    for area in areas[1:]:
        plt.text(*area.center, i, fontsize=22, ha="center", va="center", color='red')
        i += 1
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
    plt.axis([-20, 20, -20, 20])
    plt.xlabel('position x')
    plt.ylabel('position y')

    # this shows the path as an animation (although it is quite slow)
    """for i in range(len(states)):
        states_per_cycle = 1
        if i % states_per_cycle == 0 or i+1 == len(states):
            plt.gcf().clear()
            plot_areas(areas)
            plt.axis('square')
            # TODO: don't hardcode these values, or do
            plt.axis([-20, 20, -20, 20])
            plt.xlabel('position x')
            plt.ylabel('position y')
            plt.scatter(states[i, 0], states[i, 1])
            #plt.scatter(states[i+1, 0], states[i+1, 1])
            plt.pause(0.01 * states_per_cycle)"""

    # this plots the complete path:
    plt.scatter(states[:, 0], states[:, 1], c=range(len(states)), cmap=plt.cm.get_cmap('winter'))


def plot_cbf(t_end, time_delta, saved_cbfs):
    plt.figure()
    plt.locator_params(axis='x', nbins=30)
    plt.plot(np.arange(0, t_end + time_delta, time_delta)[:len(saved_cbfs)], saved_cbfs)
    plt.xlabel('time [s]')
    plt.ylabel('CBF b(x,t)')


def main():
    state = np.array([[0, 0]]).T
    saved_states = state.T
    t_end = 125

    rectangular_areas = [
        Rectangle((0, 0), 40, 40),       #  0 - Work Area
        Rectangle((-17.5, 17.5), 5, 5),  #  1 - Butter
        Rectangle((-12.5, 17.5), 5, 5),  #  2 - Sugar
        Rectangle((-7.5, 17.5), 5, 5),   #  3 - Eggs
        Rectangle((-2.5, 17.5), 5, 5),   #  4 - Flour
        Rectangle((2.5, 17.5), 5, 5),    #  5 - Baking Powder
        Rectangle((7.5, 17.5), 5, 5),    #  6 - Chocolate
        Rectangle((-17, 3), 6, 6),       #  7 - Water
        Rectangle((-17, 3), 6, 6),       #  8 - Milk
        Rectangle((-12, -16), 8, 8),     #  9 - Blenders
        Rectangle((16, 16), 8, 8),       # 10 - Ovens
        Rectangle((17, -17), 6, 6),      # 11 - Delivery Point
        Rectangle((0, -17), 6, 6),       # 12 - Starting/Ending Point
    ]

    circular_areas = [
        Rectangle((0, 0), 40, 40),   #  0 - Work Area
        Circle((-17.5, 17.5), 2.5),  #  1 - Butter
        Circle((-12.5, 17.5), 2.5),  #  2 - Sugar
        Circle((-7.5, 17.5), 2.5),   #  3 - Eggs
        Circle((-2.5, 17.5), 2.5),   #  4 - Flour
        Circle((2.5, 17.5), 2.5),    #  5 - Baking Powder
        Circle((7.5, 17.5), 2.5),    #  6 - Chocolate
        Circle((-17, 3), 3),         #  7 - Water/Milk
        Circle((-12, -16), 4),       #  8 - Mixer
        Circle((16, 16), 4),         #  9 - Ovens
        Circle((17, -17), 3),        # 10 - Delivery Point
        Circle((0, -17), 3),         # 11 - Ending Point
    ]

    alpha = 1
    time_delta = 0.2

    '''
    # for rectangular areas
    areas = rectangular_areas

    bounds_candidate = create_candidate('globally', (0, t_end), 0, 0.1, areas[0], always_active=True)
    candidate1 = create_candidate('globally', (5, 10), -25, 0.1, areas[1])
    candidate2 = create_candidate('globally', (15, 20), -65, 0.1, areas[8])
    candidate3 = create_candidate('globally', (25, 30), -200, 0.1, areas[3])
    candidate4 = create_candidate('globally', (35, 40), -280, 0.1, areas[8])
    candidate5 = create_candidate('globally', (45, 50), -360, 0.1, areas[2])
    candidate6 = create_candidate('globally', (55, 60), -440, 0.1, areas[8])
    candidate7 = create_candidate('globally', (65, 70), -390, 0.1, areas[6])
    candidate8 = create_candidate('globally', (75, 80), -450, 0.1, areas[8])
    candidate9 = create_candidate('globally', (85, 90), -510, 0.1, areas[4])
    candidate10 = create_candidate('globally', (95, 100), -570, 0.1, areas[8])
    candidate11 = create_candidate('globally', (105, 110), -630, 0.1, areas[9])
    candidate12 = create_candidate('globally', (115, 120), -690, 0.1, areas[10])
    candidate13 = create_candidate('eventually', (120, t_end), -875, 0.1, areas[11])
    cbf = bounds_candidate & candidate1 & candidate2 & candidate3 & candidate4 & candidate5 & candidate6 & candidate7 & candidate8 & candidate9 & candidate10 & candidate11 & candidate12 & candidate13
    '''

    # for circular areas
    areas = circular_areas
    
    bounds_candidate = create_candidate('globally', (0, t_end), 0, 0.1, areas[0], always_active=True)
    candidate1 = create_candidate('globally', (5, 10), -35, 0.1, areas[1])
    candidate2 = create_candidate('globally', (15, 20), -105, 0.1, areas[8])
    candidate3 = create_candidate('globally', (25, 30), -175, 0.1, areas[3])
    candidate4 = create_candidate('globally', (35, 40), -245, 0.1, areas[8])
    candidate5 = create_candidate('globally', (45, 50), -315, 0.1, areas[2])
    candidate6 = create_candidate('globally', (55, 60), -385, 0.1, areas[8])
    candidate7 = create_candidate('globally', (65, 70), -455, 0.1, areas[6])
    candidate8 = create_candidate('globally', (75, 80), -525, 0.1, areas[8])
    candidate9 = create_candidate('globally', (85, 90), -595, 0.1, areas[4])
    candidate10 = create_candidate('globally', (95, 100), -665, 0.1, areas[8])
    candidate11 = create_candidate('globally', (105, 110), -755, 0.1, areas[9])
    candidate12 = create_candidate('globally', (115, 120), -805, 0.1, areas[10])
    candidate13 = create_candidate('eventually', (120, t_end), -875, 0.1, areas[11])
    cbf = bounds_candidate & candidate1 & candidate2 & candidate3 & candidate4 & candidate5 & candidate6 & candidate7 & candidate8 & candidate9 & candidate10 & candidate11 & candidate12 & candidate13

    t = 0
    saved_cbfs = np.array([])
    while t < t_end:
        control_input = calculate_control_input(state, t, alpha, cbf)
        print(control_input.T, t)
        state = state + time_delta * control_input
        saved_states = np.append(saved_states, state.T, 0)
        saved_cbfs = np.append(saved_cbfs, cbf(state, t))
        t += time_delta

    plot_cbf(t_end, time_delta, saved_cbfs)
    #plt.axhline(y=0, color='orange', linestyle='dashed')
    plt.xlim([0, 125])
    plot_path(saved_states, areas)
    plt.show()


if __name__ == '__main__':
    main()
