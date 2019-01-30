from collections import defaultdict
import copy

import numpy as np

import treetk

class DependencyTree(object):

    def __init__(self, arcs, tokens):
        """
        :type arcs: list of (int, int, str)
        :type tokens: list of str
        """
        # NOTE that integers in arcs are not word IDs but indices in the sentence.
        self.arcs = arcs
        self.tokens = tokens

        self.head2dependents = defaultdict(list) # {int: list of (int, str)}
        self.dependent2head = {} # {int: (int/None, str/None)}

        # Create a mapping of head -> dependents
        for head, dependent, label in self.arcs:
            self.head2dependents[head].append((dependent, label))

        # Create a mapping of dependent -> head
        for dependent in range(len(self.tokens)):
            self.dependent2head[dependent] = (None, None)
        for head, dependent, label in self.arcs:
            # Tokens should not have multiple heads.
            if self.dependent2head[dependent] != (None, None):
                raise ValueError("The %d-th token has multiple heads! Arcs=%s" % (dependent, self.arcs))
            self.dependent2head[dependent] = (head, label)

    def __str__(self):
        """
        :rtype: str
        """
        return str([(str(h) + "_" + self.tokens[h], str(d) + "_" + self.tokens[d], l)
                    for h,d,l in self.arcs])

    def tolist(self, labeled=True, replace_with_tokens=False):
        """
        :type labeled: bool
        :type replace_with_tokens: bool
        :rtype: list of (T, T, str), or list of (T, T) where T \in {int, str}
        """
        result = self.arcs
        if replace_with_tokens:
            result = [(self.tokens[h], self.tokens[d], l) for h,d,l in result]
        if not labeled:
            result = [(h,d) for h,d,l in result]
        return result

    def get_dependents(self, index):
        """
        :type index: int
        :rtype: list of (int, str)
        """
        return self.head2dependents.get(index, [])

    def get_head(self, index):
        """
        :rtype index: int
        :rtype: (int, str)
        """
        return self.dependent2head[index]

def arcs2dtree(arcs, tokens=None):
    """
    :type arcs: list of (int, int, str), or list of (int, int)
    :type tokens: list of str, or None
    :rtype DependencyTree
    """
    arcs_checked = [x if len(x) == 3 else (x[0],x[1],"*") for x in arcs]
    if tokens is None:
        tokens = ["x%s" % tok_i for tok_i in range(len(arcs_checked)+1)]
    dtree = DependencyTree(arcs=arcs_checked, tokens=tokens)
    return dtree

def hyphens2arcs(hyphens):
    """
    :type hyphens: list of str
    :rtype: list of (int, int, str)
    """
    arcs = [x.split("-") for x in hyphens]
    arcs = [(int(arc[0]), int(arc[1]), str(arc[2])) if len(arc) == 3
             else (int(arc[0]), int(arc[1]), "*")
             for arc in arcs]
    return arcs

#####################################

def ctree2dtree(tree, func_label_rule):
    """
    :type NonTerminal or Terminal
    :type func_label_rule: function of (NonTerminal, int, int) -> str
    :rtype: DependencyTree
    """
    if (tree.head_token_index is None) or (tree.head_child_index is None):
        raise ValueError("Please call ``tree.calc_heads(func_head_child_rule)'' before conversion.")
    if tree.is_terminal():
        raise ValueError("``tree'' must be NonTerminal.")

    arcs = _rec_ctree2dtree(tree, func_label_rule)

    tokens = tree.leaves()
    dtree = arcs2dtree(arcs=arcs, tokens=tokens)
    return dtree

def _rec_ctree2dtree(node, func_label_rule=None):
    """
    :type node: NonTerminal or Terminal
    :type func_label_rule: function of (NonTerminal, int, int) -> str
    :rtype: list of (int, int, str)
    """
    if node.is_terminal():
        return []

    arcs = []

    # Process the child nodes
    for c_i in range(len(node.children)):
        sub_arcs = _rec_ctree2dtree(node.children[c_i], func_label_rule=func_label_rule)
        arcs.extend(sub_arcs)

    # Process the current node
    head_token_index = node.head_token_index
    for c_i in range(len(node.children)):
        dep_token_index = node.children[c_i].head_token_index
        if head_token_index == dep_token_index:
            continue
        if func_label_rule is None:
            label = "*"
        else:
            label = func_label_rule(node, node.head_child_index, c_i)
        arcs.append((head_token_index, dep_token_index, label))

    return arcs

#####################################

def dtree2ctree(dtree, binarize=None, LPAREN="(", RPAREN=")"):
    """
    :type dtree: DependencyTree
    :type binarize: None, or str
    :type LPAREN: str
    :type RPAREN: str
    :rtype: NonTerminal
    """
    # (1) Get dependency spans.
    dependency_spans = _get_dependency_spans(dtree)
    assert len(dependency_spans) == len(dtree.tokens)
    # (2) Sort. Remove spans of length 1.
    span2token = {}
    for span, token in zip(dependency_spans, dtree.tokens):
        span2token[span] = token
    dependency_spans_sorted = sorted(dependency_spans, key=lambda x: (x[0], -x[1]))
    dependency_spans_sorted_filtered = [span for span in dependency_spans_sorted if span[0] != span[1]]
    # (3) Compute left/right sides for each token.
    left_sides, right_sides = [], []
    for _ in range(len(dtree.tokens)):
        left_sides.append([])
        right_sides.append([])
    for span in dependency_spans_sorted_filtered:
        begin_i, end_i = span
        head = span2token[span] # NOTE
        left_sides[begin_i].append(LPAREN) # from left to right
        left_sides[begin_i].append(head)
        right_sides[end_i] = [RPAREN] + right_sides[end_i] # from right to left
    # (4) Create a S-expression according to the left_sides, tokens, and right_sides.
    sexp = []
    for index in range(len(dtree.tokens)):
        sexp.extend(left_sides[index])
        sexp.append(dtree.tokens[index])
        sexp.extend(right_sides[index])
    # (5) Convert the S-expression to a constituency tree instance.
    sexp = treetk.preprocess(sexp)
    ctree = treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False)
    # (6) Binarize the tree.
    if binarize is not None:
        # TODO
        pass
    return ctree

def _get_dependency_spans(dtree):
    """
    :type dtree: DependencyTree
    :rtype: list of (int, int)
    """
    # Create a map from each token to its dependents that can be traced via the arcs.
    head2dependents = _get_head2dependents_map(dtree)
    # Convert to spans.
    dependency_spans = []
    for token_index in range(len(dtree.tokens)):
        dependents = head2dependents[token_index]
        min_index = min(dependents)
        max_index = max(dependents)
        span = (min_index, max_index)
        dependency_spans.append(span)
    return dependency_spans

def _get_head2dependents_map(dtree):
    """
    :type dtree: DependencyTree
    :rtype: dictionary of {int: list of int}
    """
    head2dependents = {}

    # Add all tokens even if a token does not have any dependents.
    for token_index in range(len(dtree.tokens)):
        head2dependents[token_index] = []
        head2dependents[token_index].append(token_index)

    # Add first-order dependents.
    for head, dependent, label in dtree.arcs:
        head2dependents[head].append(dependent)

    # Add dependents recursively.
    for token_index in range(len(dtree.tokens)):
        dependents = _get_dependents_recursively(token_index, head2dependents)
        head2dependents[token_index] = dependents

    return head2dependents

def _get_dependents_recursively(head, head2dependents):
    """
    :type head: int
    :type head2dependents: dictionary of {int: list of int}
    :rtype: list of int
    """
    dependents = set(head2dependents[head])

    history = set()
    while True:
        prev_length = len(dependents)
        for dependent in copy.deepcopy(dependents):
            if dependent == head:
                continue
            if dependent in history:
                continue
            history.add(dependent)
            # Add dependents of the dependent.
            for dd in head2dependents[dependent]:
                dependents.add(dd)
        # Finish if new tokens were not added.
        new_length = len(dependents)
        if prev_length == new_length:
            break
    return list(dependents)

#####################################

LEAF_WINDOW = 8
SPACE_SIZE = 1
SPACE = " " * SPACE_SIZE

EMPTY = 0
ARROW = 1
VERTICAL = 2
HORIZONTAL = 3
# LABEL_BEGIN = 4
# LABEL_END = 5

def pretty_print_dtree(dtree, return_str=False):
    """
    :type dtree: DependencyTree
    :type return_str: bool
    :rtype: None or str
    """
    arcs_labeled = dtree.tolist(labeled=True)
    arcs_unlabeled = {(b,e) for b,e,_ in arcs_labeled}
    arc2label = {(b,e): l for b,e,l in arcs_labeled}

    # Tokens with padding
    tokens = dtree.tokens
    tokens_padded = [_pad_token(token) for token in tokens]
    # Compute heights of the arcs.
    arc2height = _get_arc2height(arcs_unlabeled)
    # Create a textmap.
    textmap = _init_textmap(tokens_padded, arc2height)
    # Edit the textmap.
    textmap = _edit_textmap(textmap, tokens_padded, arc2height, arc2label)
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

def _get_arc2height(arcs):
    """
    :type arcs: list of (int, int)
    :rtype: dictionary of {(int, int): int}
    """
    # arc2height = {(b,e): np.abs(b - e) for b, e in arcs}

    n_arcs = len(arcs)
    arcs_sorted = sorted(arcs, key=lambda x: np.abs(x[0] - x[1]))
    arc2height = {arc: 1 for arc in arcs}
    for arc_i in range(n_arcs):
        bi, ei = sorted(arcs_sorted[arc_i])
        for arc_j in range(n_arcs):
            if arc_i == arc_j:
                continue
            bj, ej = sorted(arcs_sorted[arc_j])
            if bi <= bj <= ej <= ei:
                arc2height[arcs_sorted[arc_i]] = max(arc2height[arcs_sorted[arc_j]] + 1, arc2height[arcs_sorted[arc_i]])
    return arc2height

def _init_textmap(tokens_padded, arc2height):
    """
    :type tokens_padded: list of str
    :type arc2height: dictionary of {(int, int): int}
    :rtype: numpy.ndarray(shape=(R,C), dtype="O")
    """
    max_height = -1
    for arc in arc2height.keys():
        height = arc2height[arc]
        if height > max_height:
            max_height = height
    textmap = np.zeros((1 + max_height * 2,
                        sum([len(token) for token in tokens_padded]) + (len(tokens_padded)-1) * SPACE_SIZE),
                       dtype="O")
    return textmap

def _edit_textmap(textmap, tokens_padded, arc2height, arc2label):
    """
    :type textmap: numpy.ndarray(shape=(R,C), dtype="O")
    :type tokens_padded: list of str
    :type arc2height: dictionary of {(int, int): int}
    :type arc2label: dictionary of {(int, int): str}
    :rtype: numpy.ndarray(shape=(R,C), dtype="O")
    """
    # Token index -> center position (i.e., column index in textmap)
    index2position = {} # {int: int}
    for token_i in range(len(tokens_padded)):
        center = int(len(tokens_padded[token_i]) / 2) \
                    + sum([len(token) for token in tokens_padded[:token_i]]) \
                    + SPACE_SIZE * token_i
        index2position[token_i] = center

    arcs_sorted = sorted(arc2height.keys(), key=lambda x: arc2height[x])
    for arc in arcs_sorted:
        b, e = arc
        b_pos = index2position[b]
        e_pos = index2position[e]
        height = arc2height[arc]
        label = arc2label[arc]
        # End point
        textmap[-1, e_pos] = ARROW
        textmap[-2:-1-height*2:-1, e_pos] = VERTICAL
        # Beginning point
        if b < e:
            textmap[-1, b_pos+2] = VERTICAL
        else:
            textmap[-1, b_pos-2] = VERTICAL
        # Horizontal lines
        if b < e:
            textmap[-1-height*2, b_pos+2:e_pos+1] = HORIZONTAL
        else:
            textmap[-1-height*2, e_pos:b_pos-2+1] = HORIZONTAL
        # Vertical lines
        if b < e:
            textmap[-2:-1-height*2:-1, b_pos+2] = VERTICAL
        else:
            textmap[-2:-1-height*2:-1, b_pos-2] = VERTICAL
        # Label
        if b < e:
            textmap[-1-height*2+1, e_pos-len(label):e_pos] = list(label)
        else:
            textmap[-1-height*2+1, e_pos+1:e_pos+1+len(label)] = list(label)

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
            elif textmap[row_i, col_i] == ARROW:
                row_text = row_text + "V"
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


