from .a import *


class B:
    b_op = numpy.random.rand(1000, 1000)

    def __init__(self):
        pass

    @staticmethod
    def get_sum():
        c_op = numpy.random.rand(1000, 1000)
        return add(1, 2)


def b():
    return add(1, 2) + subtract(1, 2)
