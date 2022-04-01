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


class InsideSquarePredicate:
    """A factory class for 'inside square' predicate functions."""

    def __init__(self, left, right, bottom, top):
        """The variables left, right, bottom and top are the coordinates of the square's sides."""
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

    def get_predicates(self):
        """Returns the four 'inside square' predicate functions corresponding to the four edges of the square.

        The four 'inside square' predicate functions are returned in a dictionary
        with 'left', 'right', 'bottom' and 'top' as the keys.

        The input to the predicate functions is the state variable x of the shape [position_x, position_y]."""

        def square_left_predicate(x):
            return x[0] - self.left

        def square_right_predicate(x):
            return self.right - x[0]

        def square_bottom_predicate(x):
            return x[1] - self.bottom

        def square_top_predicate(x):
            return self.top - x[1]

        return {
            'left': square_left_predicate,
            'right': square_right_predicate,
            'bottom': square_bottom_predicate,
            'top': square_top_predicate
        }

    def get_gradients(self):
        """Returns the four 'inside square' predicate functions' gradients.

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
