import numpy as np

from . import ll
from . import lu
from . import ul
from . import uu


################
# Creation
################


def sexp2tree(sexp, with_nonterminal_labels, with_terminal_labels, LPAREN="(", RPAREN=")"):
    """
    Parameters
    ----------
    sexp: list[str]
    with_nonterminal_labels: bool
    with_terminal_labels: bool
    LPAREN: str, default "("
    RPAREN: str, default ")"

    Returns
    -------
    NonTerminal
    """
    if with_nonterminal_labels and with_terminal_labels:
        tree = ll.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif with_nonterminal_labels and not with_terminal_labels:
        tree = lu.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif not with_nonterminal_labels and with_terminal_labels:
        tree = ul.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    elif not with_nonterminal_labels and not with_terminal_labels:
        tree = uu.sexp2tree(sexp, LPAREN=LPAREN, RPAREN=RPAREN)
    else:
        raise Exception("Unsupported argument pairs: with_nonterminal_labels=%s, with_terminal_labels=%s" % \
                            (with_nonterminal_labels, with_terminal_labels))
    return tree


def tree2sexp(tree):
    """
    Parameters
    ----------
    tree: NonTerminal or Terminal

    Returns
    -------
    list[str]
    """
    # sexp = tree.__str__()
    sexp = _tree2sexp(tree)
    sexp = preprocess(sexp)
    return sexp


def _tree2sexp(node):
    """
    Parameters
    ----------
    node: NonTerminal or Terminal

    Returns
    -------
    str
    """
    if node.is_terminal():
        if hasattr(node, "label"):
            return "( %s %s )" % (node.label, node.token)
        else:
            return "%s" % node.token
    else:
        inner = " ".join([_tree2sexp(c) for c in node.children])
        if hasattr(node, "label"):
            return "( %s %s )" % (node.label, inner)
        else:
            return "( %s )" % inner


def preprocess(x, LPAREN="(", RPAREN=")"):
    """
    Parameters
    ----------
    x: str or list[str]
    LPAREN: str, default "("
    RPAREN: str, default ")"

    Returns
    -------
    list[str]
    """
    if isinstance(x, list):
        x = " ".join(x)
    sexp = x.replace(LPAREN, " %s " % LPAREN).replace(RPAREN, " %s " % RPAREN).split()
    return sexp


def filter_parens(sexp, LPAREN="(", RPAREN=")"):
    """
    Parameters
    ----------
    sexp: list[str]
    LPAREN: str, default "("
    RPAREN: str, default ")"

    Returns
    -------
    list[str]
    """
    return [x for x in sexp if not x in [LPAREN, RPAREN]]


################
# Aggregation
################


def traverse(node, order="pre-order", include_terminal=True, acc=None):
    """Aggregate nodes.

    Parameters
    ----------
    node: NonTerminal or Terminal
    order: str, default "pre-order"
    include_terminal: bool, default True
    acc: list[T] or None, where T denotes NonTerminal or Terminal, default None

    Returns
    -------
    list[T], where T denotes NonTerminal or Terminal
    """
    if acc is None:
        acc = []

    if node.is_terminal():
        if include_terminal:
            acc.append(node)
        return acc

    if order == "pre-order":
        # Process the current node
        acc.append(node)
        # Process the child nodes
        for c in node.children:
            acc = traverse(c, order=order, include_terminal=include_terminal, acc=acc)
    elif order == "post-order":
        # Process the child nodes
        for c in node.children:
            acc = traverse(c, order=order, include_terminal=include_terminal, acc=acc)
        # Process the current node
        acc.append(node)
    else:
        raise ValueError("Invalid order: %s" % order)

    return acc


def aggregate_production_rules(root, order="pre-order", include_terminal=True):
    """Aggregate production rules.

    Parameters
    ----------
    root: NonTerminal
    order: str, default "pre-order"
    include_terminal: bool, default True

    Returns
    -------
    list[tuple[str]]

    Examples
    --------
    >>> sexp = "(S (NP (DT a) (NN cat)) (VP (VBZ bites) (NP (DT a) (NN mouse))))"
    >>> ctree = treetk.sexp2tree(treetk.preprocess(sexp))
    >>> treetk.aggregate_production_rules(ctree, include_terminal=True, order="pre-order")
	[('S', 'NP', 'VP'),
	 ('NP', 'DT', 'NN'),
	 ('DT', 'a'),
	 ('NN', 'cat'),
	 ('VP', 'VBZ', 'NP'),
	 ('VBZ', 'bites'),
	 ('NP', 'DT', 'NN'),
	 ('DT', 'a'),
	 ('NN', 'mouse')]
    """
    # NOTE: only for trees with nonterminal labels
    assert root.with_nonterminal_labels
    if include_terminal:
        assert root.with_terminal_labels

    nodes = traverse(root, order=order, include_terminal=include_terminal, acc=None)

    rules = []
    for node in nodes:
        if node.is_terminal():
            # Terminal node
            if node.with_terminal_labels:
                rules.append((node.label, node.token))
        else:
            # Non-Terminal node
            if node.with_terminal_labels:
                # e.g., NP -> DT NN
                rhs = [c.label for c in node.children]
            else:
                # e.g., NP -> a mouse
                rhs = [c.token if c.is_terminal() else c.label for c in node.children]
            rule = [node.label] + list(rhs)
            rules.append(tuple(rule))
    return rules


def aggregate_spans(root, include_terminal=False, order="pre-order"):
    """Aggregate spans.

    Parameters
    ----------
    root: NonTerminal or Terminal
    include_terminal: bool, default False
    order: str, default "pre-order"

    Returns
    -------
    list[(int,int,str)] or list[(int,int)]

    Examples
    --------
    >>> sexp = "(S (NP (DT a) (NN cat)) (VP (VBZ bites) (NP (DT a) (NN mouse))))"
    >>> ctree = treetk.sexp2tree(treetk.preprocess(sexp))
    >>> treetk.aggregate_spans(ctree, include_terminal=True, order="pre-order")
    [(0, 4, 'S')
     (0, 1, 'NP')
     (0, 0, 'DT')
     (1, 1, 'NN')
     (2, 4, 'VP')
     (2, 2, 'VBZ')
     (3, 4, 'NP')
     (3, 3, 'DT')
     (4, 4, 'NN')]
    """
    nodes = traverse(root, order=order, include_terminal=include_terminal, acc=None)

    spans = []
    for node in nodes:
        if node.is_terminal():
            if node.with_terminal_labels:
                # e.g., (2, 2, "NN")
                spans.append(tuple(list(node.index_span) + [node.label]))
            else:
                # e.g., (2, 2)
                spans.append(node.index_span)
        else:
            if node.with_nonterminal_labels:
                # e.g., (2, 4, "NP")
                spans.append(tuple(list(node.index_span) + [node.label]))
            else:
                # e.g., (2, 4)
                spans.append(node.index_span)

    return spans


def aggregate_composition_spans(root, order="pre-order", binary=True):
    """Aggregate composed span pairs

    Parameters
    ----------
    root: NonTerminal or Terminal
    order: str, default "pre-order"
    binary: bool, default True

    Returns
    -------
    list[[(int,int), (int,int), str]] or list[[(int,int), (int,int)]]

     Examples
    --------
    >>> sexp = "(S (NP (DT a) (NN cat)) (VP (VBZ bites) (NP (DT a) (NN mouse))))"
    >>> ctree = treetk.sexp2tree(treetk.preprocess(sexp))
    >>> treetk.aggregate_composition_spans(ctree, order="pre-order", binary=True)
	[[(0, 1), (2, 4), 'S'],
	 [(0, 0), (1, 1), 'NP'],
	 [(2, 2), (3, 4), 'VP'],
	 [(3, 3), (4, 4), 'NP']]
    """
    nodes = traverse(root, order=order, include_terminal=False, acc=None)

    # Check
    if binary:
        for node in nodes:
            assert len(node.children) == 2

    comp_spans = []
    for node in nodes:
        if node.with_nonterminal_labels:
            # e.g., [(0,1), (2,4), "NP"]
            comp_spans.append([c.index_span for c in node.children] + [node.label])
        else:
            # e.g., [(0,1), (2,4)]
            comp_spans.append([c.index_span for c in node.children])

    return comp_spans


def aggregate_constituents(root, order="pre-order"):
    """Aggregate constituents.

    Parameters
    ----------
    root: NonTerminal or Terminal
    order: str, default "pre-order"

    Returns
    -------
    list[list[str]]

    Examples
    --------
    >>> sexp = "(S (NP (DT a) (NN cat)) (VP (VBZ bites) (NP (DT a) (NN mouse))))"
    >>> ctree = treetk.sexp2tree(treetk.preprocess(sexp))
    >>> treetk.aggregate_constituents(ctree, order="pre-order")
	[['a', 'cat', 'bites', 'a', 'mouse'],
	 ['a', 'cat'],
	 ['bites', 'a', 'mouse'],
	 ['a', 'mouse']]
    """
    nodes = traverse(root, order=order, include_terminal=False, acc=None)

    constituents = []
    for node in nodes:
        constituents.append(node.leaves())

    return constituents


################
# Tree modification
################


def left_shift(node):
    """
    Parameters
    ----------
    node: NonTerminal

    Returns
    -------
    NonTerminal

    Notes
    --------
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
    Parameters
    ----------
    node: NonTerminal

    Returns
    -------
    NonTerminal

    Notes
    -----
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
# Label assignment
################


def assign_labels(node, span2label, with_terminal_labels):
    """
    Parameters
    ----------
    node: NonTerminal or Terminal
    span2label: dict[(int, int), str]
    with_terminal_labels: bool

    Returns
    -------
    NonTerminal or Terminal
    """
    if node.is_terminal():
        if with_terminal_labels:
            # Terminal
            assert node.index_span in span2label
            node.label = span2label[node.index_span]
            node.with_terminal_labels = True
        else:
            pass
    else:
        # NonTerminal
        assert node.index_span in span2label
        node.label = span2label[node.index_span]
        node.with_nonterminal_labels = True
    if not node.is_terminal():
        for c_i in range(len(node.children)):
            node.children[c_i] = assign_labels(node.children[c_i], span2label, with_terminal_labels=with_terminal_labels)
    return node


################
# Checking
################


def is_completely_binary(node):
    """
    Parameters
    ----------
    node: NonTerminal or Terminal

    Returns
    -------
    bool
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
################


LEAF_WINDOW = 8
SPACE_SIZE = 1
SPACE = " " * SPACE_SIZE

EMPTY = 0
VERTICAL = 1
HORIZONTAL = 2


def pretty_print(tree, return_str=False, LPAREN="(", RPAREN=")", separate_leaves=False):
    """
    Parameters
    ----------
    tree: NonTerminal or Terminal
    return_str: bool, default False
    LPAREN: str, default "("
    RPAREN: str, default ")"

    Returns
    -------
    None or str
    """
    # Tokens with padding
    tokens = tree.leaves()
    tokens_padded = [_pad_token(token, separate_leaves=separate_leaves) for token in tokens]
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


def _pad_token(token, separate_leaves):
    """
    Parameters
    ----------
    token: str

    Returns
    -------
    str
    """
    token = " " + token + " "
    while len(token) <= LEAF_WINDOW:
        token = " " + token + " "
    if separate_leaves:
        token = "[" + token[1:-1] + "]"
    return token


def _init_textmap(tokens_padded, tree):
    """
    Parameters
    ----------
    tokens_padded: list[str]
    tree: NonTerminal or Terminal

    Returns
    -------
    numpy.ndarray(shape=(R,C), dtype="O")
    """
    max_height = tree.set_height()
    max_height += 1 # include POS nodes
    textmap = np.zeros((max_height * 3,
                        sum([len(token) for token in tokens_padded]) + (len(tokens_padded)-1) * SPACE_SIZE),
                       dtype="O")
    return textmap


def _edit_textmap(textmap, tokens_padded, tree):
    """
    Parameters
    ----------
    textmap: numpy.ndarray(shape=(R,C), dtype="O")
    tokens_padded: list[str]
    tree: NonTerminal or Terminal

    Returns
    -------
    numpy.ndarray(shape=(R,C), dtype="O")
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
    Parameters
    ----------
    node: NonTerminal or Terminal
    index2position: dict[int, int]

    Returns
    --------
    numpy.ndarray(shape=(R,C), dtype=int)
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
    Parameters
    ----------
    node: NonTerminal or Terminal
    textmap: numpy.ndarray(shape=(R,C), dtype=int)

    Returns
    -------
    numpy.ndarray(shape=(R,C), dtype=int)
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
    Parameters
    ----------
    node: NonTerminal or Terminal
    textmap: numpy.ndarray(shape=(R,C), dtype="O")

    Returns
    -------
    numpy.ndarray(shape=(R,C), dtype="O")
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
    Parameters
    ----------
    textmap: numpy.ndarray(shape=(R,C), dtype="O")
    tokens_padded: list[str]

    Returns
    -------
    str
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
    Parameters
    ----------
    tree: NonTerminal or Terminal
    LPAREN: str, default "("
    RPAREN: str, default ")"
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
    Parameters
    ----------
    tree: NonTerminal or Terminal
    LPAREN: str, default "("
    RPAREN: str, default ")"
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
    Parameters
    ----------
    text: str
    with_terminal_labels: bool
    LPAREN: str, default "("

    Returns
    -------
    str
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


