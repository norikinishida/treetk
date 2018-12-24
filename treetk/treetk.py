import sys

import numpy as np

################
# Conversion (sexp -> tree, or tree -> sexp)

def sexp2tree(sexp, with_nonterminal_labels, with_terminal_labels, LPAREN="(", RPAREN=")"):
    """
    :type sexp: list of str
    :type with_nonterminal_labels: bool
    :type with_terminal_labels: bool
    :type LPAREN: str
    :type RPAREN: str
    :rtype: NonTerminal
    """
    if with_nonterminal_labels and with_terminal_labels:
        from . import full
        tree = full.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif with_nonterminal_labels and not with_terminal_labels:
        from . import partial
        tree = partial.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif not with_nonterminal_labels and with_terminal_labels:
        from . import partial2
        tree = partial2.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif not with_nonterminal_labels and not with_terminal_labels:
        from . import leavesonly
        tree = leavesonly.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    else:
        print("Unsupported argument pairs: with_nonterminal_labels=%s, with_terminal_labels=%s" % \
                (with_nonterminal_labels, with_terminal_labels))
        sys.exit(-1)
    return tree

def tree2sexp(tree):
    """
    :type tree: NonTerminal or Terminal
    :rtype: list of str
    """
    sexp = tree.__str__()
    sexp = preprocess(sexp)
    return sexp

def preprocess(x, LPAREN="(", RPAREN=")"):
    """
    :type x: str or list of str
    :rtype: list of str
    """
    if isinstance(x, list):
        x = " ".join(x)
    sexp = x.replace(LPAREN, " %s " % LPAREN).replace(RPAREN, " %s " % RPAREN).split()
    return sexp

def filter_parens(sexp, LPAREN="(", RPAREN=")"):
    """
    :type sexp: list of str
    :type LPAREN: str
    :type RPAREN: str
    :rtype: list of str
    """
    return [x for x in sexp if not x in [LPAREN, RPAREN]]

################
# Aggregation of production rules
# NOTE: only for trees with nonterminal labels

def aggregate_production_rules(root):
    """
    :type root: NonTerminal
    :rtype: list of tuple of str
    """
    assert root.with_nonterminal_labels
    nodes = _rec_aggregate_production_rules(root)
    return nodes

def _rec_aggregate_production_rules(node, acc=None):
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
            acc = _rec_aggregate_production_rules(c, acc=acc)

    return acc

################
# Aggregation of spans

def aggregate_spans(node, acc=None):
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
        acc.append(tuple([node.label] + list(node.index_span)))
    else:
        acc.append(node.index_span)

    for c in node.children:
        acc = aggregate_spans(c, acc=acc)
    return acc

def aggregate_composition_spans(node, acc=None, binary=True):
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
        # acc.append([node.label, node.children[0].index_span, node.children[1].index_span])
        acc.append([node.label] + [c.index_span for c in node.children])
    else:
        # acc.append([node.children[0].index_span, node.children[1].index_span])
        acc.append([c.index_span for c in node.children])

    for c in node.children:
        acc = aggregate_composition_spans(c, acc=acc, binary=binary)

    return acc

################
# Aggregation of subtrees.

def aggregate_subtrees(root, string=True):
    """
    :type root: NonTerminal
    :type string: bool
    :rtype: list of str
    """
    nodes = _rec_aggregate_subtrees(root)
    if string:
        nodes = [n.__str__() for n in nodes]
    return nodes

def _rec_aggregate_subtrees(node, acc=None):
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
            acc = _rec_aggregate_subtrees(c, acc=acc)

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

def right_shift(node):
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
# Checking

def is_completely_binary(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: bool
    """
    if node.is_terminal():
        return True
    if len(node.children) != 2:
        return False
    acc = True
    for c in node.children:
        acc *= is_completely_binary(c)
    return bool(acc)

################
# Visualization

LEAF_WINDOW = 8
SPACE_SIZE = 1
SPACE = " " * SPACE_SIZE

EMPTY = 0
VERTICAL = 1
HORIZONTAL = 2

def pretty_print(tree, return_str=False, LPAREN="(", RPAREN=")"):
    """
    :type tree: NonTerminal or Terminal
    :type LPAREN: str
    :type RPAREN: str
    :rtype: None or str
    """
    # Tokens with padding
    tokens = tree.leaves()
    tokens_padded = [_pad_token(token) for token in tokens]
    # Create a textmap.
    textmap = _init_textmap(tokens_padded, tree)
    # Edit the textmap.
    textmap = _edit_textmap(textmap, tokens_padded, tree)
    # Create a text based on the textmap.
    text = _generate_text(textmap, tokens_padded)

    if return_str:
        return text
    else:
        print(text)

def _pad_token(token):
    """
    :type token: str
    :rtype: str
    """
    token = " " + token + " "
    while len(token) <= LEAF_WINDOW:
        token = " " + token + " "
    token = "[" + token[1:-1] + "]"
    return token

def _init_textmap(tokens_padded, tree):
    """
    :type tokens_padded: list of str
    :type tree: NonTerminal or Terminal
    :rtype: numpy.ndarray(shape=(R,C), dtype="O")
    """
    max_height = tree.set_height()
    max_height += 1 # include POS nodes
    textmap = np.zeros((max_height * 3,
                        sum([len(token) for token in tokens_padded]) + (len(tokens_padded)-1) * SPACE_SIZE),
                       dtype="O")
    return textmap

def _edit_textmap(textmap, tokens_padded, tree):
    """
    :type textmap: numpy.ndarray(shape=(R,C), dtype="O")
    :type tokens_padded: list of str
    :type tree: NonTerminal or Terminal
    :rtype: numpy.ndarray(shape=(R,C), dtype="O")
    """
    # Token index -> center position (i.e., column index in textmap)
    index2position = {} # {int: int}
    for token_i in range(len(tokens_padded)):
        center = int(len(tokens_padded[token_i]) / 2) \
                    + sum([len(token) for token in tokens_padded[:token_i]]) \
                    + SPACE_SIZE * token_i
        index2position[token_i] = center

    # Edit
    tree = _set_position_for_each_node(tree, index2position)
    textmap = _edit_horizontal_lines(tree, textmap)
    textmap = _edit_vertical_lines(tree, textmap)

    # Reverse and post-processing
    textmap = textmap[::-1, :]
    textmap = textmap[1:, :]
    return textmap

def _set_position_for_each_node(node, index2position):
    """
    :type node: NonTerminal or Terminal
    :type index2position: {int: int}
    :rtype: numpy.ndarray(shape=(R,C), dtype=int)
    """
    if node.is_terminal():
        position = index2position[node.index]
        node.position = position
        return node

    min_position = np.inf
    max_position = -np.inf
    for c in node.children:
        c = _set_position_for_each_node(c, index2position)
        if c.position < min_position:
            min_position = c.position
        if c.position > max_position:
            max_position = c.position

    position = (min_position + max_position) // 2
    node.position = position

    return node

def _edit_vertical_lines(node, textmap):
    """
    :type node: NonTerminal or Terminal
    :type textmap: numpy.ndarray(shape=(R,C), dtype=int)
    :rtype: numpy.ndarray(shape=(R,C), dtype=int)
    """
    row_i = node.height * 3 + 1
    col_i = node.position

    textmap[row_i-1, col_i] = VERTICAL
    textmap[row_i+1, col_i] = VERTICAL

    if node.with_nonterminal_labels and not node.is_terminal():
        label = list(node.label)
    elif node.with_terminal_labels and node.is_terminal():
        label = list(node.label)
    else:
        label = "*"

    if len(label) % 2 == 0:
        half = len(label) // 2 - 1
    else:
        half = len(label) // 2
    former_label = label[0:half]
    latter_label = label[half:]

    if (col_i - len(former_label) < 0) or (textmap.shape[1] < col_i + len(latter_label)):
        raise Exception("Node label='%s' is too long. Please use treetk.nltk_pretty_print() instead." % node.label)

    textmap[row_i, col_i-len(former_label):col_i] = former_label
    textmap[row_i, col_i:col_i+len(latter_label)] = latter_label

    if node.is_terminal():
        return textmap

    max_height = -1
    for c in node.children:
        if c.height > max_height:
            max_height = c.height
    for c in node.children:
        textmap[(c.height * 3 + 1) + 2: (max_height * 3 + 1) + 2, c.position] = VERTICAL

    for c in node.children:
        textmap = _edit_vertical_lines(c, textmap)

    return textmap

def _edit_horizontal_lines(node, textmap):
    """
    :type node: NonTerminal or Terminal
    :type textmap: numpy.ndarray(shape=(R,C), dtype="O")
    :rtype: numpy.ndarray(shape=(R,C), dtype="O")
    """
    if node.is_terminal():
        return textmap

    min_position = np.inf
    max_position = -np.inf
    for c in node.children:
        if c.position < min_position:
            min_position = c.position
        if c.position > max_position:
            max_position = c.position

    row_i = node.height * 3 + 1 - 1
    left_col_i = min_position
    right_col_i = max_position

    textmap[row_i, left_col_i:right_col_i + 1] = HORIZONTAL

    for c in node.children:
        textmap = _edit_horizontal_lines(c, textmap)

    return textmap

def _generate_text(textmap, tokens_padded):
    """
    :type textmap: numpy.ndarray(shape=(R,C), dtype="O")
    :type tokens_padded: list of str
    """
    text = ""
    for row_i in range(textmap.shape[0]):
        row_text = ""
        for col_i in range(textmap.shape[1]):
            if textmap[row_i, col_i] == EMPTY:
                row_text = row_text + " "
            elif textmap[row_i, col_i] == VERTICAL:
                row_text = row_text + "|"
            elif textmap[row_i, col_i] == HORIZONTAL:
                row_text = row_text + "_"
            else:
                row_text = row_text + str(textmap[row_i, col_i])
        row_text = row_text.rstrip() + "\n"
        text = text + row_text
    for token in tokens_padded:
        text = text + token
        text = text + SPACE
    text = text[:-SPACE_SIZE]
    return text

def nltk_pretty_print(tree, LPAREN="(", RPAREN=")"):
    """
    :type tree: NonTerminal or Terminal
    :type LPAREN: str
    :type RPAREN: str
    :rtype: None
    """
    import nltk.tree
    text = tree.__str__()
    if not tree.with_nonterminal_labels:
        text = _insert_dummy_nonterminal_labels(text,
                with_terminal_labels=tree.with_terminal_labels,
                LPAREN=LPAREN)
    nltk.tree.Tree.fromstring(text).pretty_print()

def nltk_draw(tree, LPAREN="(", RPAREN=")"):
    """
    :type tree: NonTerminal or Terminal
    :type LPAREN: str
    :type RPAREN: str
    :rtype: None
    """
    import nltk.tree
    text = tree.__str__()
    if not tree.with_nonterminal_labels:
        text = _insert_dummy_nonterminal_labels(text,
                with_terminal_labels=tree.with_terminal_labels,
                LPAREN=LPAREN)
    nltk.tree.Tree.fromstring(text).draw()

def _insert_dummy_nonterminal_labels(text, with_terminal_labels, LPAREN="("):
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


