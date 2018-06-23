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
    elif not with_nonterminal_labels and not with_terminal_labels:
        import leavesonly
        tree = leavesonly.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    else:
        print("Unsupported argument pairs: with_nonterminal_labels=%s, with_terminal_labels=%s" % \
                (with_nonterminal_labels, with_terminal_labels))
        sys.exit(-1)
    return tree

################
# 前処理
def preprocess(x, LPAREN="(", RPAREN=")"):
    """
    :type x: str or list of str
    :rtype: list of str
    """
    if isinstance(x, list):
        x = " ".join(x)
    sexp = x.replace(LPAREN, " %s " % LPAREN).replace(RPAREN, " %s " % RPAREN).split()
    return sexp

################
# rangesの収集 e.g., {(i, j)}, or {[(i,k), (k+1,j)]}
def aggregate_ranges(node, acc=None, with_nonterminal_labels=False):
    """
    :type node: NonTerminal or Terminal
    :type acc: list of (int,int), or list of (str,int,int), or None
    :type with_nonterminal_labels: bool
    :rtype: list of (int,int), or list of (str,int,int)
    """
    if acc is None:
        acc = []
    if node.is_terminal():
        return acc
    if with_nonterminal_labels:
        acc.append(tuple([node.label] + list(node.index_range)))
    else:
        acc.append(node.index_range)
    for c in node.children:
        acc = aggregate_ranges(c, acc=acc, with_nonterminal_labels=with_nonterminal_labels)
    return acc

def aggregate_merging_ranges(node, acc=None, with_nonterminal_labels=False):
    """
    :type node: NonTerminal or Terminal
    :type acc: list of [(int,int), (int,int)], or list of [str, (int,int), (int,int)], or None
    :type with_nonterminal_labels: bool
    :rtype: list of [(int,int), (int,int)], or list of [str, (int,int), (int,int)]
    """
    if acc is None:
        acc = []
    if node.is_terminal():
        return acc
    assert len(node.children) == 2
    if with_nonterminal_labels:
        acc.append([node.label, node.children[0].index_range, node.children[1].index_range])
    else:
        acc.append([node.children[0].index_range, node.children[1].index_range])
    for c in node.children:
        acc = aggregate_merging_ranges(c, acc=acc, with_nonterminal_labels=with_nonterminal_labels)
    return acc

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
    if not tree.with_nonterminal_labels and not tree.with_terminal_labels:
        text = text.replace(LPAREN, "%s * " % LPAREN)
    nltk.tree.Tree.fromstring(text).pretty_print()

def draw(tree, LPAREN="(", RPAREN=")"):
    """
    :type tree: NonTerminal or Terminal
    :type LPAREN: str
    :type RPAREN: str
    :rtype: None
    """
    text = tree.__str__()
    if not tree.with_nonterminal_labels and not tree.with_terminal_labels:
        text = text.replace(LPAREN, "%s * " % LPAREN)
        text = text.replace(LPAREN, "%s * " % LPAREN)
    nltk.tree.Tree.fromstring(text).draw()

