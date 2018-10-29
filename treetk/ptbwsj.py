# -*- coding: utf-8 -*-

from full import NonTerminal, Terminal

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
    if node.is_terminal():
        node.token = node.token.lower()
        return node
    for c_i in range(len(node.children)):
        node.children[c_i] = lowercasing(node.children[c_i])
    return node

def remove_empties_and_punctuations(node):
    node = _remove_empties_and_punctuations_1(node)
    node = _remove_empties_and_punctuations_2(node)
    return node

def _remove_punctuations(node):
    if node.is_terminal():
        return node
    new_children = []
    for c_i in range(len(node.children)):
        # 子ノードで, 終端ノードかつpunctuationであるものは除去
        if node.children[c_i].is_terminal() and \
                node.children[c_i].label in ["-NONE-", ",", ".", ":", "``", "''", "-LRB-", "-RRB-", "$", "#"]:
            continue
        new_children.append(node.children[c_i])
    node.children = new_children
    for c_i in range(len(node.children)):
        node.children[c_i] = _remove_punctuations(node.children[c_i])
    return node

def _remove_empties(node):
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
    if node.is_terminal():
        return 1
    count = 0
    for c in node.children:
        count += _count_terminals(c)
    return count

############################
# 非終端ノードラベルの前処理

def remove_function_tags(node):
    if node.is_terminal():
        return node
    node.label = _remove_function_tags(node.label)
    for c_i in range(len(node.children)):
        node.children[c_i] = remove_function_tags(node.children[c_i])
    return node

def _remove_function_tags(label):
    if "-" in label and not label in ["-NONE-", "-LRB-", "-RRB-"]:
        lst = label.split("-")
        return lst[0]
    else:
        return label

############################
# 二分木化

def binarize(node):
    if node.is_terminal():
        return node
    if len(node.children) > 2:
        node.children = _right_branching(node.children)
    for c_i in range(len(node.children)):
        node.children[c_i] = binarize(node.children[c_i])
    return node

def _right_branching(nodes):
    if len(nodes) == 2:
        return nodes
    else:
        lhs = nodes[0]
        rhs = NonTerminal("+".join([n.label for n in nodes[1:]]))
        rhs.children = _right_branching(nodes[1:])
        return [lhs, rhs]

############################
# その他

# def add_dummy_node(node):
#     if node.is_terminal():
#         return node
#     if len(node.children) == 1:
#         new_node = Terminal(label="<DUMMY>", token="___")
#         node.add_child(new_node)
#     for c_i in range(len(node.children)):
#         node.children[c_i] = add_dummy_node(node.children[c_i])
#     return node


