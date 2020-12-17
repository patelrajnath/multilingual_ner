from typing import Iterable, Tuple, Union, List


class Token(object):
    def __init__(self, text, idx, i):
        self.text = text
        self.idx = idx
        self.i = i

    def __str__(self):
        return self.text
