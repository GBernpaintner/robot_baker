import numpy as np


class LinearGamma:
    """A factory class for linear gamma functions."""

    def __init__(self, gamma_0, gamma_inf, t_end):
        """The variable t_end changes depending on the type of task:
        t_end = b for eventually
        t_end = a for globally."""

        self.gamma_0 = gamma_0
        self.gamma_inf = gamma_inf
        self.t_end = t_end

    def get_function(self):
        """Returns the linear gamma function gamma(t)."""

        def gamma(t):
            if t < self.t_end:
                return (self.gamma_inf - self.gamma_0) / self.t_end * t + self.gamma_0
            else:
                return self.gamma_inf

        return gamma

    def get_gradient(self):
        """Returns the gradient of the gamma function gamma_gradient(t)."""

        def gamma_gradient(t):
            if t < self.t_end:
                return (self.gamma_inf - self.gamma_0) / self.t_end
            else:
                return 0

        return gamma_gradient


# TODO ensure the predicate gradients never divide by 0


class InsideCirclePredicate:
    """A factory class for 'inside circle' predicate functions."""

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def get_predicates(self):
        """Returns the 'inside circle' predicate function circle_predicate(x) in a list.

        The input to the predicate function is the state variable x of the shape [position_x, position_y]."""

        def circle_predicate(x):
            return self.radius - np.linalg.norm(self.center - x)

        return [circle_predicate]

    def get_gradients(self):
        """Returns the gradient of the predicate function circle_predicate_gradient(x) in a list.

        The gradient takes the state x as input and expects it to be a column vector."""

        def circle_predicate_gradient(x):
            gradient_x = x[0, 0] - self.center[0] / (np.linalg.norm(np.array([self.center]).T - x) + 0.00001)
            gradient_y = x[1, 0] - self.center[1] / (np.linalg.norm(np.array([self.center]).T - x) + 0.00001)
            return np.array([[gradient_x, gradient_y]]).T

        return [circle_predicate_gradient]


class InsideRectanglePredicate:
    """A factory class for 'inside rectangle' predicate functions."""

    def __init__(self, center, width, height):
        """The variable center are the coordinates (x, y) of the rectangle's center."""
        self.center = center
        self.width = width
        self.height = height

    def get_predicates(self):
        """Returns the two 'inside rectangle' predicate functions corresponding to the x- and y-bounds.

        The two 'inside rectangle' predicate functions are returned in a list like so: [x_gradient, y_gradient].

        The input to the predicate functions is the state variable x in the shape [position_x, position_y]."""

        def x_predicate(x):
            return self.width / 2 - abs(x[0] - self.center[0])

        def y_predicate(x):
            return self.height / 2 - abs(x[1] - self.center[1])

        return [x_predicate, y_predicate]

    def get_gradients(self):
        """Returns the two 'inside rectangle' predicate functions' gradients.

        The two gradients are returned in a list like so: [x_gradient, y_gradient].

        The input to the gradients is the state variable x,
        a numpy.ndarray, of the shape [[position_x, position_y]].T.

        The output is a numpy column vector (numpy.ndarray) with two rows."""

        def x_predicate_gradient(x):
            x_gradient = -(x[0, 0] - self.center[0]) / (abs(x[0, 0] - self.center[0]) + 0.00001)
            y_gradient = 0
            return np.array([[x_gradient, y_gradient]]).T

        def y_predicate_gradient(x):
            x_gradient = 0
            y_gradient = -(x[1, 0] - self.center[1]) / (abs(x[1, 0] - self.center[1]) + 0.00001)
            return np.array([[x_gradient, y_gradient]]).T

        return [x_predicate_gradient, y_predicate_gradient]
