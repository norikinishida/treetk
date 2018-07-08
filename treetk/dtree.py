# -*- coding: utf-8 -*-

from collections import defaultdict
import copy

import treetk

class DependencyTree:

    def __init__(self, tokens, arcs, labels):
        """
        :type tokens: list of str
        :type arcs: list of (int, int)
        :type labels: list of str
        """
        self.tokens = tokens
        self.arcs = arcs
        self.labels = labels

        self.head2dependents = self.create_head2dependents()

    def create_head2dependents(self):
        """
        :rtype: dictionary of {int: list of (int,str)}
        """
        # NOTE: 再帰的にまではdependentsを辿らないことに注意
        dictionary = defaultdict(list)
        for (head, dependent), label in zip(self.arcs, self.labels):
            dictionary[head].append((dependent, label))
        return dictionary

    def __str__(self):
        """
        :rtype: str
        """
        return str([(str(h) + "_" + self.tokens[h], str(d) + "_" + self.tokens[d], l)
                    for (h,d),l in zip(self.arcs, self.labels)])

    def tolist(self, labeled=True, replace_with_tokens=False):
        """
        :type labeled: bool
        :type replace_with_tokens: bool
        :rtype: list of (T, T, str), or list of (T, T) where T \in {int, str}
        """
        result = self.arcs
        if replace_with_tokens:
            result = [(self.tokens[h], self.tokens[d]) for h,d in result]
        if labeled:
            result = [(h,d,l) for (h,d),l in zip(result, self.labels)]
        return result

    def todict(self, labeled=True, replace_with_tokens=False):
        """
        :type labeled: bool
        :type replace_with_tokens: bool
        :rtype: dictionary of {T: list of (T, str)}, or dictionary of {T: list of T} where T \in {int, str}
        """
        if labeled:
            result = dict(self.head2dependents)
            if replace_with_tokens:
                result = {self.tokens[h]: [(self.tokens[d], l) for d,l in values]
                          for h,values in result.items()}
        else:
            result = {h: [d for d,l in values] for h,values in self.head2dependents.items()}
            if replace_with_tokens:
                result = {self.tokens[h]: [self.tokens[d] for d in ds] for h,ds in result.items()}
        return result

def produce_dependencytree(tokens, arcs, labels=None):
    """
    :type tokens: list of str
    :type arcs: list of (int, int)
    :type labels: None, or list of str
    :rtype DependencyTree
    """
    if labels is None:
        labels = ["*" for _ in range(len(arcs))]
    else:
        assert len(labels) == len(arcs)
    dtree = DependencyTree(tokens=tokens, arcs=arcs, labels=labels)
    return dtree

def ctree2dtree(tree):
    """
    :type NonTerminal or Terminal
    :rtype: DependencyTree
    """
    # TODO
    pass

def dtree2ctree(dtree, binarize=None, LPAREN="(", RPAREN=")"):
    """
    :type dtree: DependencyTree
    :type binarize: None, or str
    :type LPAREN: str
    :type RPAREN: str
    :rtype: NonTerminal
    """
    # (1) dependency spansの取得
    dependency_spans = _get_dependency_spans(dtree)
    assert len(dependency_spans) == len(dtree.tokens)
    # (2) ソート, length=1のspanの除去
    span2token = {}
    for span, token in zip(dependency_spans, dtree.tokens):
        span2token[span] = token
    dependency_spans_sorted = sorted(dependency_spans, key=lambda x: (x[0], -x[1]))
    dependency_spans_sorted_filtered = [span for span in dependency_spans_sorted if span[0] != span[1]]
    # (3) 各トークンについて，Left/Right sidesを計算
    left_sides, right_sides = [], []
    for _ in range(len(dtree.tokens)):
        left_sides.append([])
        right_sides.append([])
    for span in dependency_spans_sorted_filtered:
        begin_i, end_i = span
        head = "<" + span2token[span] + ">" # NOTE
        left_sides[begin_i].append(LPAREN) # from left to right
        left_sides[begin_i].append(head)
        right_sides[end_i] = [RPAREN] + right_sides[end_i] # from right to left
    # (4) left_sides, tokens, right_sidesに従ってS式を作成
    sexp = []
    for index in range(len(dtree.tokens)):
        sexp.extend(left_sides[index])
        sexp.append(dtree.tokens[index])
        sexp.extend(right_sides[index])
    # (5) treeの作成
    sexp = treetk.preprocess(sexp)
    ctree = treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False)
    # (6) binarization?
    if binarize is not None:
        # TODO
        pass
    return ctree

def _get_dependency_spans(dtree):
    """
    :type dtree: DependencyTree
    :rtype: list of (int, int)
    """
    # 各トークンからlinkで辿れるすべてのdependentsへのマップ
    head2dependents = _get_head2dependents_map(dtree)
    # spanへの変換
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
    head2dependents = defaultdict(list)

    # 自分自身も含む (dependentsを持たない単語も登録する)
    for token_index in range(len(dtree.tokens)):
        head2dependents[token_index].append(token_index)

    # 1次のdependentsを登録
    for head, dependent in dtree.arcs:
        head2dependents[head].append(dependent)

    # 再帰的にdependentsを登録
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
            # 一度見たものはもう見ない
            if dependent in history:
                continue
            history.add(dependent)
            # dependentのdependentsを登録
            for dd in head2dependents[dependent]:
                dependents.add(dd)
        # 新しいdependentが登録されてなければ終了
        new_length = len(dependents)
        if prev_length == new_length:
            break
    return list(dependents)

