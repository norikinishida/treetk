# -*- coding: utf-8 -*-

import numpy as np

class Terminal(object):
    def __init__(self, token, index):
        self.token = token  
        self.index = index
        self.index_range = (index, index)

    def leaves(self):
        return [self.token]

    def calc_ranges(self):
        return self.index_range

    def is_terminal(self):
        return True

    def tolist(self):
        return self.token

    def __str__(self):
        return "%s" % self.token

class NonTerminal(object):
    def __init__(self):
        self.children = []
        self.index_range = (None, None)

    def add_child(self, node):
        self.children.append(node)

    def leaves(self):
        leaves = []
        for c in self.children:
            leaves.extend(c.leaves())
        return leaves

    def calc_ranges(self):
        min_index = np.inf
        max_index = -np.inf
        for c_i in xrange(len(self.children)):
            i, j = self.children[c_i].calc_ranges()
            if i < min_index:
                min_index = i
            if max_index < j:
                max_index = j
        self.index_range = (min_index, max_index)
        return min_index, max_index

    def is_terminal(self):
        return False

    def tolist(self):
        inner = [c.tolist() for c in self.children]
        return inner

    def __str__(self):
        inner = " ".join([c.__str__() for c in self.children])
        return "( %s )" % inner

