import math

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
        self.head_token_index = index
        self.head_child_index = index

        self.depth = None
        self.height = None

        self.with_terminal_labels = False
        self.with_nonterminal_labels = False

    ###########
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

    ###########
    def is_terminal(self):
        """
        :rtype: bool
        """
        return True

    ###########
    def calc_spans(self):
        """
        :rtype: (int, int)
        """
        return self.index_span

    def calc_heads(self, func_head_child_rule):
        """
        :type func_head_child_rule: function of NonTerminal -> int
        :rtype: int
        """
        return self.head_token_index

    ###########
    def set_depth(self, depth=0):
        """
        :type depth: int
        :rtype: int
        """
        self.depth = depth
        return self.depth

    def set_height(self):
        """
        :rtype: int
        """
        self.height = 0
        return self.height

class NonTerminal(object):

    def __init__(self):
        """
        :rtype: None
        """
        self.children = []

        self.index_span = (None, None)
        self.head_token_index = None
        self.head_child_index = None

        self.depth = None
        self.height = None

        self.with_terminal_labels = False
        self.with_nonterminal_labels = False

    def add_child(self, node):
        """
        :type node: NonTerminal or Terminal
        :rtype: None
        """
        self.children.append(node)

    ###########
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

    ###########
    def is_terminal(self):
        """
        :rtype: bool
        """
        return False

    ###########
    def calc_spans(self):
        """
        :rtype: (int, int)
        """
        min_index = math.inf
        max_index = -math.inf
        for c_i in range(len(self.children)):
            i, j = self.children[c_i].calc_spans()
            if i < min_index:
                min_index = i
            if max_index < j:
                max_index = j
        self.index_span = (min_index, max_index)
        return min_index, max_index

    def calc_heads(self, func_head_child_rule):
        """
        :type func_head_child_rule: function of NonTerminal -> int
        :rtype: int
        """
        head_token_indices = []
        for c_i in range(len(self.children)):
            head_token_index = self.children[c_i].calc_heads(func_head_child_rule)
            head_token_indices.append(head_token_index)

        self.head_child_index = func_head_child_rule(self)
        self.head_token_index = head_token_indices[self.head_child_index]
        return self.head_token_index

    ###########
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

    def set_height(self):
        """
        :rtype: int
        """
        max_height = -1
        for c_i in range(len(self.children)):
            cheight = self.children[c_i].set_height()
            if cheight > max_height:
                max_height = cheight
        self.height = max_height + 1
        return self.height

def sexp2tree(sexp, LPAREN, RPAREN):
    """
    :type sexp: list of str, e.g. "( ( a cat ) ( bites ( a mouse ) ) )".split()
    :type LPAREN: str
    :type RPAREN: str
    :rtype: NonTerminal
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


