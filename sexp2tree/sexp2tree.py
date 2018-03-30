# -*- coding: utf-8 -*-

def sexp2tree(sexp, pattern=0, LPAREN="(", RPAREN=")"):
    if pattern == 0:
        return sexp2tree_pattern_0(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif pattern == 1:
        return sexp2tree_pattern_1(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif pattern == 2:
        return sexp2tree_pattern_2(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    else:
        import sys
        sys.exit(-1)

def sexp2tree_pattern_0(sexp, LPAREN, RPAREN):
    """
    IN: tokenized bracket string, e.g., "( ( a cat ) ( bites ( a mouse ) ) )".split()
    OUT: NonTerminal 
    NOTE: prepare_ptb_wsj.pyのsexp2tree()とは入力に想定しているS式の仕様が異なることに注意
    """
    from nodes_pattern_0 import Terminal, NonTerminal
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

def sexp2tree_pattern_1(sexp, LPAREN, RPAREN):
    """
    IN: tokenized bracket string, e.g., "( S ( NP a cat ) ( VP bites ( NP a mouse ) ) )".split()
    OUT: NonTerminal
    """
    from nodes_pattern_1 import Terminal, NonTerminal
    tokens = sexp
    n_tokens = len(tokens)
    i = 0
    pos_count = 0
    ROOT = NonTerminal("ROOT")
    stack = [ROOT]
    while i < n_tokens:
        if tokens[i] == LPAREN:
            node = NonTerminal(label=tokens[i+1]) # XXX
            stack.append(node)
            i += 2
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

def sexp2tree_pattern_2(sexp, LPAREN, RPAREN):
    """
    IN: tokenized bracket string, e.g., "( S ( NP ( DT a ) ( NN cat ) ) ( VP ( VBZ bites ) ( NP ( DT a ) ( NN mouse ) ) ) )".split()
    OUT: NonTerminal
    """
    from nodes_pattern_2 import Terminal, NonTerminal
    tokens = sexp
    n_tokens = len(tokens)
    i = 0
    pos_count = 0
    ROOT = NonTerminal("ROOT")
    stack = [ROOT]
    while i < n_tokens:
        if tokens[i] == LPAREN:
            assert tokens[i+1] not in [LPAREN, RPAREN]
            node = NonTerminal(label=tokens[i+1]) # XXX
            stack.append(node)
            i += 2
        elif tokens[i] == RPAREN:
            node = stack.pop()
            stack[-1].add_child(node)
            i += 1
        else:
            # 終端ノードだと思ってプッシュしたけど非終端ノードだった
            node = stack.pop()
            node = Terminal(label=node.label, token=tokens[i], index=pos_count)
            pos_count += 1
            stack.append(node)
            i += 1
    assert len(stack) == 1
    ROOT = stack.pop()
    assert len(ROOT.children) == 1
    return ROOT.children[0]

def aggregate_ranges(node, acc=[]):
    if node.is_terminal():
        return acc
    acc.append(node.index_range)
    for c in node.children:
        acc = aggregate_ranges(c, acc=acc)
    return acc

def aggregate_merging_ranges(node, acc=[]):
    if node.is_terminal():
        return acc
    assert len(node.children) == 2
    acc.append([node.children[0].index_range, node.children[1].index_range])
    for c in node.children:
        acc = aggregate_merging_ranges(c, acc=acc)
    return acc
