class Rectangle:
    def __init__(self, center, width, height):
        self.center = center
        self.width = width
        self.height = height
        self.bottom_left = (center[0] - width / 2, center[1] - height / 2)


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
