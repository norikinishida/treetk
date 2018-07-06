# -*- coding: utf-8 -*-

import numpy as np

class Terminal(object):
    def __init__(self, token, index):
        """
        :type token: str
        :type index: int
        :rtype: None
        """
        self.token = token
        self.index = index
        self.index_span = (index, index)
        self.depth = None
        self.with_terminal_labels = False
        self.with_nonterminal_labels = False

    def __str__(self):
        """
        :rtype: str
        """
        return "%s" % self.token

    def tolist(self):
        """
        :rtype: str
        """
        return self.token

    def leaves(self):
        """
        :rtype: list of str, i.e., [str]
        """
        return [self.token]

    def is_terminal(self):
        """
        :rtype: bool
        """
        return True

    def calc_spans(self):
        """
        :rtype: (int, int)
        """
        return self.index_span

    def set_depth(self, depth=0):
        """
        :type depth: int
        :rtype: int
        """
        self.depth = depth
        return self.depth

class NonTerminal(object):
    def __init__(self):
        """
        :rtype: None
        """
        self.children = []
        self.index_span = (None, None)
        self.depth = None
        self.with_terminal_labels = False
        self.with_nonterminal_labels = False

    def add_child(self, node):
        """
        :type node: NonTerminal or Terminal
        :rtype: None
        """
        self.children.append(node)

    def __str__(self):
        """
        :rtype: str
        """
        inner = " ".join([c.__str__() for c in self.children])
        return "( %s )" % inner

    def tolist(self):
        """
        :rtype: (list of)+ str
        """
        inner = [c.tolist() for c in self.children]
        return inner

    def leaves(self):
        """
        :rtype: list of str
        """
        leaves = []
        for c in self.children:
            leaves.extend(c.leaves())
        return leaves

    def is_terminal(self):
        """
        :rtype: bool
        """
        return False

    def calc_spans(self):
        """
        :rtype: (int, int)
        """
        min_index = np.inf
        max_index = -np.inf
        for c_i in range(len(self.children)):
            i, j = self.children[c_i].calc_spans()
            if i < min_index:
                min_index = i
            if max_index < j:
                max_index = j
        self.index_span = (min_index, max_index)
        return min_index, max_index

    def set_depth(self, depth=0):
        """
        :type depth: int
        :rtype: int
        """
        self.depth = depth
        max_cdepth = -1
        for c_i in range(len(self.children)):
            cdepth = self.children[c_i].set_depth(depth=depth+1)
            if cdepth > max_cdepth:
                max_cdepth = cdepth
        return max_cdepth

def sexp2tree(sexp, LPAREN, RPAREN):
    """
    :type sexp: list of str, e.g. "( ( a cat ) ( bites ( a mouse ) ) )".split()
    :type LPAREN: str
    :type RPAREN: str
    :rtype: NonTerminal
    NOTE: prepare_ptb_wsj.pyのsexp2tree()とは入力に想定しているS式の仕様が異なることに注意
    """
    tokens = sexp
    n_tokens = len(tokens)
    i = 0
    pos_count = 0
    ROOT = NonTerminal()
    stack = [ROOT]
    while i < n_tokens:
        if tokens[i] == LPAREN:
            node = NonTerminal()
            stack.append(node)
            i += 1
        elif tokens[i] == RPAREN:
            node = stack.pop()
            stack[-1].add_child(node)
            i += 1
        else:
            node = Terminal(token=tokens[i], index=pos_count)
            pos_count += 1
            stack[-1].add_child(node)
            i += 1
    assert len(stack) == 1
    ROOT = stack.pop()
    assert len(ROOT.children) == 1
    return ROOT.children[0]


