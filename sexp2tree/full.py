# -*- coding: utf-8 -*-

import numpy as np

class Terminal(object):
    def __init__(self, label, token, index):
        """
        :type label: str
        :type token: str
        :type index: int
        :rtype: None
        """
        self.label = label
        self.token = token
        self.index = index
        self.index_range = (index, index)
        self.with_nonterminal_labels = True
        self.with_terminal_labels = True

    def calc_ranges(self):
        """
        :rtype: (int, int)
        """
        return self.index_range

    def is_terminal(self):
        """
        :rtype: bool
        """
        return True

    def __str__(self):
        """
        :rtype: str
        """
        return "( %s %s )" % (self.label, self.token)

    def tolist(self):
        """
        :rtype: list of str, i.e., [str, str]
        """
        return [self.label, self.token]

    def leaves(self):
        """
        :rtype: list of str, i.e, [str]
        """
        return [self.token]

class NonTerminal(object):
    def __init__(self, label):
        """
        :type label: str
        :rtype: None
        """
        self.label = label
        self.children = []
        self.index_range = (None, None)
        self.with_nonterminal_labels = True
        self.with_terminal_labels = True

    def add_child(self, node):
        """
        :type node: NonTerminal or Terminal
        :rtype: None
        """
        self.children.append(node)

    def calc_ranges(self):
        """
        :rtype: (int, int)
        """
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

    def __str__(self):
        """
        :rtype: str
        """
        inner = " ".join([c.__str__() for c in self.children])
        return "( %s %s )" % (self.label, inner)

    def tolist(self):
        """
        :rtype: (list of)+ str
        """
        inner = [self.label] + [c.tolist() for c in self.children]
        return inner

    def leaves(self):
        """
        :rtype: list of str
        """
        leaves = []
        for c in self.children:
            leaves.extend(c.leaves())
        return leaves



def sexp2tree(sexp, LPAREN, RPAREN):
    """
    :type sexp: list of str, e.g., "( S ( NP ( DT a ) ( NN cat ) ) ( VP ( VBZ bites ) ( NP ( DT a ) ( NN mouse ) ) ) )".split()
    :type LPAREN: str
    :type RPAREN: str
    :rtype: NonTerminal
    """
    tokens = sexp
    n_tokens = len(tokens)
    i = 0
    pos_count = 0
    ROOT = NonTerminal("ROOT")
    stack = [ROOT]
    while i < n_tokens:
        if tokens[i] == LPAREN:
            assert tokens[i+1] not in [LPAREN, RPAREN]
            node = NonTerminal(label=tokens[i+1])
            stack.append(node)
            i += 2
        elif tokens[i] == RPAREN:
            node = stack.pop()
            stack[-1].add_child(node)
            i += 1
        else:
            # 非終端ノードだと思ってプッシュしたけど終端ノードだった
            node = stack.pop()
            node = Terminal(label=node.label, token=tokens[i], index=pos_count)
            pos_count += 1
            stack.append(node)
            i += 1
    assert len(stack) == 1
    ROOT = stack.pop()
    assert len(ROOT.children) == 1
    return ROOT.children[0]


