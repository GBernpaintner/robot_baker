from math import log, exp

import numpy as np

from factory_classes import InsideCirclePredicate, InsideRectanglePredicate
from areas import Circle, Rectangle


class InsideTask:
    """Defines a task specifying an area to stay within."""
    def __init__(self, area, gamma):
        self.gamma = gamma
        if isinstance(area, Circle):
            self.predicate_factory = InsideCirclePredicate(area.center, area.radius)
        elif isinstance(area, Rectangle):
            self.predicate_factory = InsideRectanglePredicate(area.center, area.width, area.height)

        self.barriers = self.generate_barriers()
        self.barrier_dbdxs = self.predicate_factory.get_gradients()
        self.barrier_dbdt = gamma.get_gradient()

    def deactivation_policy(self, t):
        return 1 if t <= self.gamma.t_end else 0

    def generate_barriers(self):
        """Generates the barrier functions corresponding to the task."""
        barriers = []
        for predicate in self.predicate_factory.get_predicates():
            def make_barrier(predicate):
                def barrier(x, t):
                    return -self.gamma.get_function()(t) + predicate(x)
                return barrier
            barriers.append(make_barrier(predicate))
        return barriers


class TotalBarrier:
    def __init__(self, tasks):
        self.tasks = tasks

    def evaluate(self, x, t):
        total_barrier_sum = 0
        for task in self.tasks:
            for barrier in task.barriers:
                total_barrier_sum += task.deactivation_policy(t) * exp(-barrier(x, t))
        return -log(total_barrier_sum)

    def evaluate_dbdx(self, x, t):
        total_numerator_sum = np.array([[0., 0.]]).T
        total_denominator_sum = 0
        for task in self.tasks:
            for i in range(len(task.barriers)):
                barrier = task.barriers[i]
                dbdx = task.barrier_dbdxs[i]
                total_numerator_sum += task.deactivation_policy(t) * exp(-barrier(x, t)) * dbdx(x)
                total_denominator_sum += task.deactivation_policy(t) * exp(-barrier(x, t))
        return total_numerator_sum / total_denominator_sum

    def evaluate_dbdt(self, x, t):
        total_numerator_sum = 0
        total_denominator_sum = 0
        for task in self.tasks:
            for i in range(len(task.barriers)):
                barrier = task.barriers[i]
                total_numerator_sum += task.deactivation_policy(t) * exp(-barrier(x, t)) * task.barrier_dbdt(t)
                total_denominator_sum += task.deactivation_policy(t) * exp(-barrier(x, t))
        return total_numerator_sum / total_denominator_sum
