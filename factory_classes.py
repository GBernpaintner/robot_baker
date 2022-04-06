import numpy as np


class LinearGamma:
    """A factory class for linear gamma functions."""

    def __init__(self, gamma_0, gamma_inf, t_end):
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


class InsideCirclePredicate:
    """A factory class for 'inside circle' predicate functions."""

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def get_predicate(self):
        """Returns the 'inside circle' predicate function circle_predicate(x).

        The input to the predicate function is the state variable x of the shape [position_x, position_y]."""

        def circle_predicate(x):
            return self.radius - np.linalg.norm(self.center - x)

        return circle_predicate

    def get_gradient(self):
        """Returns the gradient of the predicate function circle_predicate_gradient(x).

        The gradient takes the state x as input and expects it to be a column vector."""

        def circle_predicate_gradient(x):
            gradient_x = x[0] - self.center[0] / np.linalg.norm(self.center - x)
            gradient_y = x[1] - self.center[1] / np.linalg.norm(self.center - x)
            return np.array([[gradient_x, gradient_y]]).T

        return circle_predicate_gradient


class InsideRectanglePredicate:
    """A factory class for 'inside rectangle' predicate functions."""

    def __init__(self, center, width, height):
        """The variable center are the coordinates (x, y) of the rectangle's center."""
        self.center = center
        self.width = width
        self.height = height

    def get_predicates(self):
        """Returns the four 'inside rectangle' predicate functions corresponding to the four edges of the rectangle.

        The four 'inside rectangle' predicate functions are returned in a dictionary
        with 'left', 'right', 'bottom' and 'top' as the keys.

        The input to the predicate functions is the state variable x in the shape [position_x, position_y]."""

        def square_left_predicate(x):
            left = self.center[0] - self.width / 2
            return x[0] - left

        def square_right_predicate(x):
            right = self.center[0] + self.width / 2
            return right - x[0]

        def square_bottom_predicate(x):
            bottom = self.center[1] - self.height
            return x[1] - bottom

        def square_top_predicate(x):
            top = self.center[1] + self.height
            return top - x[1]

        return {
            'left': square_left_predicate,
            'right': square_right_predicate,
            'bottom': square_bottom_predicate,
            'top': square_top_predicate
        }

    def get_gradients(self):
        """Returns the four 'inside rectangle' predicate functions' gradients.

        The four gradients are returned in a dictionary with 'left', 'right', 'bottom' and 'top' as the keys.

        The input to the gradients is the state variable x of the shape [position_x, position_y].

        The output is a numpy column vector (numpy.ndarray) with two rows."""

        def square_left_predicate_gradient(x):
            return np.array([[1, 0]]).T

        def square_right_predicate_gradient(x):
            return np.array([[-1, 0]]).T

        def square_bottom_predicate_gradient(x):
            return np.array([[0, 1]]).T

        def square_top_predicate_gradient(x):
            return np.array([[0, -1]]).T

        return {
            'left': square_left_predicate_gradient,
            'right': square_right_predicate_gradient,
            'bottom': square_bottom_predicate_gradient,
            'top': square_top_predicate_gradient
        }
