# -*- coding: utf-8 -*-

import sys

import nltk.tree

def sexp2tree(sexp, with_nonterminal_labels=False, with_terminal_labels=False, LPAREN="(", RPAREN=")"):
    """
    :type sexp: list of str
    :type with_nonterminal_labels: bool
    :type with_terminal_labels: bool
    :type LPAREN: str
    :type RPAREN: str
    :rtype: NonTerminal
    """
    if with_nonterminal_labels and with_terminal_labels:
        import full
        tree = full.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif with_nonterminal_labels and not with_terminal_labels:
        import partial
        tree = partial.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif not with_nonterminal_labels and with_terminal_labels:
        import partial2
        tree = partial2.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif not with_nonterminal_labels and not with_terminal_labels:
        import leavesonly
        tree = leavesonly.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    else:
        print("Unsupported argument pairs: with_nonterminal_labels=%s, with_terminal_labels=%s" % \
                (with_nonterminal_labels, with_terminal_labels))
        sys.exit(-1)
    return tree

################
# 生成規則の収集

def aggregate_production_rules(root):
    """
    :type root: NonTerminal
    :rtype: list of tuple of str
    """
    assert root.with_nonterminal_labels
    nodes = rec_aggregate_production_rules(root)
    return nodes

def rec_aggregate_production_rules(node, acc=None):
    """
    :type node: NonTerminal or Terminal
    :type acc: list of tuple of str, or None
    :rtype: list of tuple of str
    """
    if acc is None:
        acc = []

    if node.is_terminal():
        if node.with_terminal_labels:
            acc.append((node.label, node.token))
    else:
        if node.with_terminal_labels:
            # e.g., NP -> DT NN
            rhs = [c.label for c in node.children]
        else:
            # e.g., NP -> a mouse
            rhs = [c.token if c.is_terminal() else c.label for c in node.children]
        rule = [node.label] + list(rhs)
        acc.append(tuple(rule))

    if not node.is_terminal():
        for c in node.children:
            acc = rec_aggregate_production_rules(c, acc=acc)

    return acc

################
# rangesの収集 e.g., {(i, j)}, or {[(i,k), (k+1,j)]}

def aggregate_ranges(node, acc=None):
    """
    :type node: NonTerminal or Terminal
    :type acc: list of (int,int), or list of (str,int,int), or None
    :rtype: list of (int,int), or list of (str,int,int)
    """
    if acc is None:
        acc = []

    if node.is_terminal():
        return acc

    if node.with_nonterminal_labels:
        acc.append(tuple([node.label] + list(node.index_range)))
    else:
        acc.append(node.index_range)

    for c in node.children:
        acc = aggregate_ranges(c, acc=acc)
    return acc

def aggregate_composition_ranges(node, acc=None, binary=True):
    """
    :type node: NonTerminal or Terminal
    :type acc: list of [(int,int,...), (int,int,...)], or list of [str, (int,int,...), (int,int,...)], or None
    :type binary: bool
    :rtype: list of [(int,int,...), (int,int,...)], or list of [str, (int,int,...), (int,int,...)]
    """
    if acc is None:
        acc = []

    if node.is_terminal():
        return acc

    if binary:
        # Check
        if len(node.children) != 2:
            raise ValueError("(A nonterminal node does NOT have two children. The node is %s" % node)

    if node.with_nonterminal_labels:
        # acc.append([node.label, node.children[0].index_range, node.children[1].index_range])
        acc.append([node.label] + [c.index_range for c in node.children])
    else:
        # acc.append([node.children[0].index_range, node.children[1].index_range])
        acc.append([c.index_range for c in node.children])

    for c in node.children:
        acc = aggregate_composition_ranges(c, acc=acc, binary=binary)

    return acc

################
# 部分木リストの収集 c.f., subtree kernel

def aggregate_subtrees(root, string):
    """
    :type root: NonTerminal
    :type string: bool
    :rtype: list of str
    """
    nodes = rec_aggregate_subtrees(root)
    if string:
        nodes = [n.__str__() for n in nodes]
    return nodes

def rec_aggregate_subtrees(node, acc=None):
    """
    :type node: NonTerminal or Terminal
    :type acc: list of (NonTerminal or Terminal), or None
    :rtype: list of (NonTerminal or Terminal)
    """
    if acc is None:
        acc = []

    if node.is_terminal() and (not node.with_terminal_labels):
        pass
    else:
        acc.append(node)

    if not node.is_terminal():
        for c in node.children:
            acc = rec_aggregate_subtrees(c, acc=acc)

    return acc

################
# Tree shifting

def left_shift(node):
    """
    :type node: NonTerminal
    :rtype: NonTerminal

    e.g., (A (B C)) -> ((A B) C)
    """
    assert not node.is_terminal()
    assert len(node.children) == 2
    assert not node.children[1].is_terminal()
    right = node.children[1]
    node.children[1] = None
    tmp = right.children[0]
    right.children[0] = None
    node.children[1] = tmp
    right.children[0] = node
    return right

def  right_shift(node):
    """
    :type node: NonTerminal
    :rtype: NonTerminal

    e.g., ((A B) C) -> (A (B C))
    """
    assert not node.is_terminal()
    assert len(node.children) == 2
    assert not node.children[0].is_terminal()
    left = node.children[0]
    node.children[0] = None
    tmp = left.children[1]
    left.children[1] = None
    node.children[0] = tmp
    left.children[1] = node
    return left

################
# その他の処理

def preprocess(x, LPAREN="(", RPAREN=")"):
    """
    :type x: str or list of str
    :rtype: list of str
    """
    if isinstance(x, list):
        x = " ".join(x)
    sexp = x.replace(LPAREN, " %s " % LPAREN).replace(RPAREN, " %s " % RPAREN).split()
    return sexp

def filter_parens(sexp, PARENS):
    """
    :type sexp: list of str
    :type PARENS: list of str, e.g., ["(", ")"]
    :rtype: list of str
    """
    return [x for x in sexp if not x in PARENS]

def tree2sexp(tree):
    """
    :type tree: NonTerminal or Terminal
    :rtype: list of str
    """
    sexp = tree.__str__()
    sexp = preprocess(sexp)
    return sexp

################
# チェック

def check_whether_completely_binary(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: int (1 or 0)
    """
    if node.is_terminal():
        return 1
    if len(node.children) != 2:
        return 0
    acc = 1
    for c in node.children:
        acc *= check_whether_completely_binary(c)
    return acc

################
# 描画周り

def pretty_print(tree, LPAREN="(", RPAREN=")"):
    """
    :type tree: NonTerminal or Terminal
    :type LPAREN: str
    :type RPAREN: str
    :rtype: None
    """
    text = tree.__str__()
    if not tree.with_nonterminal_labels:
        text = insert_dummy_nonterminal_labels(text,
                with_terminal_labels=tree.with_terminal_labels,
                LPAREN=LPAREN)
    nltk.tree.Tree.fromstring(text).pretty_print()

def draw(tree, LPAREN="(", RPAREN=")"):
    """
    :type tree: NonTerminal or Terminal
    :type LPAREN: str
    :type RPAREN: str
    :rtype: None
    """
    text = tree.__str__()
    if not tree.with_nonterminal_labels:
        text = insert_dummy_nonterminal_labels(text,
                with_terminal_labels=tree.with_terminal_labels,
                LPAREN=LPAREN)
    nltk.tree.Tree.fromstring(text).draw()

def insert_dummy_nonterminal_labels(text, with_terminal_labels, LPAREN="("):
    """
    :type text: str
    :type with_terminal_labels: bool
    :rtype: str
    """
    if not with_terminal_labels:
        text = text.replace(LPAREN, "%s * " % LPAREN)
    else:
        sexp = preprocess(text)
        for i in range(len(sexp)-1):
            if (sexp[i] == LPAREN) and (sexp[i+1] == LPAREN):
                sexp[i] = "%s * " % LPAREN
        text = " ".join(sexp)
    return text


