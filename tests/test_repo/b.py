from .a import *


class B:
    def __init__(self):
        pass

    @staticmethod
    def get_sum():
        return add(1, 2)

def b():
    return add(1, 2) + subtract(1, 2)
