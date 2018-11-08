# -*- coding: utf-8 -*-

from full import NonTerminal
# from full import Terminal

############################
# 読み込み

def read_sexps(path, LPAREN="(", RPAREN=")"):
    """
    :type path: str
    :rtype: list of list of str
    """
    sexps = []
    buf = []
    depth = 0
    for line in open(path):
        tokens = line.decode("utf-8").strip().replace("(", " ( ").replace(")", " ) ").split()
        if len(tokens) == 0:
            continue
        for token in tokens:
            if token == LPAREN:
                depth += 1
            elif token == RPAREN:
                depth -= 1
            buf.append(token)
            if depth == 0:
                # sexps.append(buf)
                sexps.append(buf[1:-1]) # Remove the outermost parens
                buf = []
    return sexps

############################
# トークンの前処理

def lowercasing(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: NonTerminal or Terminal
    """
    if node.is_terminal():
        node.token = node.token.lower()
        return node
    for c_i in range(len(node.children)):
        node.children[c_i] = lowercasing(node.children[c_i])
    return node

def remove_punctuations_and_empties(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: NonTerminal or Terminal
    """
    node = _remove_punctuations(node)
    node = _remove_empties(node)
    return node

def _remove_punctuations(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: NonTerminal or Terminal
    """
    if node.is_terminal():
        return node
    new_children = []
    for c_i in range(len(node.children)):
        # 子ノードで, 終端ノードかつpunctuationであるものは除去
        if node.children[c_i].is_terminal() and \
                node.children[c_i].label in ["-NONE-",
                                             ",", ".", "?", "!", "...",
                                             ":", ";",
                                             "``", "''",
                                             "--", "-",
                                             "-LRB-", "-RRB-",
                                             "-LCB-", "-RCB-",
                                             "$", "#"]:
            continue
        new_children.append(node.children[c_i])
    node.children = new_children
    for c_i in range(len(node.children)):
        node.children[c_i] = _remove_punctuations(node.children[c_i])
    return node

def _remove_empties(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: NonTerminal or Terminal
    """
    if node.is_terminal():
        return node
    new_children = []
    for c_i in range(len(node.children)):
        # 終端ノードを一つも子に持っていない非終端ノードを除去
        if _count_terminals(node.children[c_i]) > 0:
            new_children.append(node.children[c_i])
    node.children = new_children
    for c_i in range(len(node.children)):
        node.children[c_i] = _remove_empties(node.children[c_i])
    return node

def _count_terminals(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: int
    """
    if node.is_terminal():
        return 1
    count = 0
    for c in node.children:
        count += _count_terminals(c)
    return count

############################
# 非終端ノードラベルの前処理

def remove_function_tags(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: NonTerminal or Terminal
    """
    if node.is_terminal():
        return node
    node.label = _remove_function_tags(node.label)
    for c_i in range(len(node.children)):
        node.children[c_i] = remove_function_tags(node.children[c_i])
    return node

def _remove_function_tags(label):
    """
    :type label: str
    :rtype: str
    """
    if "-" in label and not label in ["-NONE-", "-LRB-", "-RRB-", "-LCB-", "-RCB-"]:
        lst = label.split("-")
        return lst[0]
    else:
        return label

############################
# 二分木化

def binarize(node, right_branching=True):
    """
    :type node: NonTerminal or Terminal
    :type right_branching: bool
    :rtype: NonTerminal or Terminal
    """
    if node.is_terminal():
        return node
    if len(node.children) > 2:
        if right_branching:
            node.children = _right_branching(node.children)
        else:
            node.children = _left_branching(node.children)
    for c_i in range(len(node.children)):
        node.children[c_i] = binarize(node.children[c_i])
    return node

def _right_branching(nodes):
    """
    :type node: NonTerminal or Terminal
    :rtype: [NonTerminal/Terminal, NonTerminal/Terminal]
    """
    if len(nodes) == 2:
        return nodes
    else:
        lhs = nodes[0]
        rhs = NonTerminal("+".join([n.label for n in nodes[1:]]))
        rhs.children = _right_branching(nodes[1:])
        return [lhs, rhs]

def _left_branching(nodes):
    """
    :type node: NonTerminal or Terminal
    :rtype: [NonTerminal/Terminal, NonTerminal/Terminal]
    """
    if len(nodes) == 2:
        return nodes
    else:
        lhs = NonTerminal("+".join([n.label for n in nodes[:-1]]))
        lhs.children = _left_branching(nodes[:-1])
        rhs = nodes[-1]
        return [lhs, rhs]

############################
# その他

# def add_dummy_node(node):
#     """
#     :type node: NonTerminal or Terminal
#     :rtype: NonTerminal or Terminal
#     """
#     if node.is_terminal():
#         return node
#     if len(node.children) == 1:
#         new_node = Terminal(label="<DUMMY>", token="___")
#         node.add_child(new_node)
#     for c_i in range(len(node.children)):
#         node.children[c_i] = add_dummy_node(node.children[c_i])
#     return node


