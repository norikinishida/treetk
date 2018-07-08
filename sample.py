# -*- coding: utf-8 -*-

import treetk

DRAW=False

def print_list(msg, xs):
    print(msg)
    for x in xs:
        print("\t%s" % str(x))

# Labels for both non-terminals and terminals
print("#############")
sexp = treetk.preprocess("(S (NP (DT a) (NN cat)) (VP (VBZ bites) (NP (DT a) (NN mouse))))")
print("sexp = %s" % sexp)

tree = treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=True)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].tolist() = %s" % tree.children[0].tolist())
print("tree.children[1].tolist() = %s" % tree.children[1].tolist())
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())

treetk.pretty_print(tree)

rules = treetk.aggregate_production_rules(tree)
print_list("production rules =", rules)

tree.calc_spans()
spans = treetk.aggregate_spans(tree)
print_list("spans =", spans)
mrg_spans = treetk.aggregate_composition_spans(tree)
print_list("composition spans =", mrg_spans)

subtree_strings = treetk.aggregate_subtrees(tree, string=True)
print_list("subtrees =", subtree_strings)

print("tree2sexp(tree) = %s" % treetk.tree2sexp(tree))

if DRAW:
    treetk.draw(tree)

# Labels only for non-terminals, but no labels for terminals
print("#############")
sexp = treetk.preprocess("(S (NP a cat) (VP bites (NP a mouse)))")
print("sexp = %s" % sexp)

tree = treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].tolist() = %s" % tree.children[0].tolist())
print("tree.children[1].tolist() = %s" % tree.children[1].tolist())
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())

treetk.pretty_print(tree)

rules = treetk.aggregate_production_rules(tree)
print_list("production rules =", rules)

tree.calc_spans()
spans = treetk.aggregate_spans(tree)
print_list("spans =", spans)
mrg_spans = treetk.aggregate_composition_spans(tree)
print_list("composition spans =", mrg_spans)

subtree_strings = treetk.aggregate_subtrees(tree, string=True)
print_list("subtrees =", subtree_strings)

print("tree2sexp(tree) = %s" % treetk.tree2sexp(tree))

if DRAW:
    treetk.draw(tree)

# No labels for non-terminals, but only labels for terminals
print("#############")
sexp = treetk.preprocess("(((DT a) (NN cat)) ((VBZ bites) ((DT a) (NN mouse))))")
print("sexp = %s" % sexp)

tree = treetk.sexp2tree(sexp, with_nonterminal_labels=False, with_terminal_labels=True)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].tolist() = %s" % tree.children[0].tolist())
print("tree.children[1].tolist() = %s" % tree.children[1].tolist())
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())

treetk.pretty_print(tree)

tree.calc_spans()
spans = treetk.aggregate_spans(tree)
print_list("spans =", spans)
mrg_spans = treetk.aggregate_composition_spans(tree)
print_list("composition spans =", mrg_spans)

subtree_strings = treetk.aggregate_subtrees(tree, string=True)
print_list("subtrees =", subtree_strings)

print("tree2sexp(tree) = %s" % treetk.tree2sexp(tree))

if DRAW:
    treetk.draw(tree)

# No labels for non-terminals and terminals
print("#############")
sexp = treetk.preprocess("((a cat) (bites (a mouse)))")
print("sexp = %s" % sexp)

tree = treetk.sexp2tree(sexp, with_nonterminal_labels=False, with_terminal_labels=False)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].tolist() = %s" % tree.children[0].tolist())
print("tree.children[1].tolist() = %s" % tree.children[1].tolist())
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())

treetk.pretty_print(tree)

tree.calc_spans()
spans = treetk.aggregate_spans(tree)
print_list("spans =", spans)
mrg_spans = treetk.aggregate_composition_spans(tree)
print_list("composition spans =", mrg_spans)

subtree_strings = treetk.aggregate_subtrees(tree, string=True)
print_list("subtrees =", subtree_strings)

print("tree2sexp(tree) = %s" % treetk.tree2sexp(tree))

if DRAW:
    treetk.draw(tree)

# Can handle unary or n-ary trees
print("#############")
sexp = treetk.preprocess("(NP (NP (NP (N w0)) (NP (N w1))) (NP (N w2) (N w3) (N w4)))")
print("sexp = %s" % sexp)

tree = treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=True)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].tolist() = %s" % tree.children[0].tolist())
print("tree.children[1].tolist() = %s" % tree.children[1].tolist())
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())

treetk.pretty_print(tree)

rules = treetk.aggregate_production_rules(tree)
print_list("production rules =", rules)

tree.calc_spans()
spans = treetk.aggregate_spans(tree)
print_list("spans =", spans)
mrg_spans = treetk.aggregate_composition_spans(tree, binary=False)
print_list("composition spans =", mrg_spans)

subtree_strings = treetk.aggregate_subtrees(tree, string=True)
print_list("subtrees =", subtree_strings)

print("tree2sexp(tree) = %s" % treetk.tree2sexp(tree))

if DRAW:
    treetk.draw(tree)

# Dependency trees can also be processed
print("#############")
tokens = ["a", "boy", "saw", "a", "girl", "with", "a", "telescope", "<$>"]
arcs = [(1, 0), (2, 1), (2, 4), (4, 3), (2, 5), (5, 7), (7, 6), (8, 2)]
labels = ["det", "nsubj", "dobj", "det", "prep", "pobj", "det", "root"]
print("tokens = %s" % tokens)
print("arcs = %s" % arcs)
print("labels = %s" % labels)
dtree = treetk.produce_dependencytree(tokens=tokens, arcs=arcs, labels=labels)
print("dtree.__str__() = %s" % dtree)
print("dtree.tolist(labeled=True) = %s" % dtree.tolist(labeled=True))
print("dtree.tolist(labeled=False) = %s" % dtree.tolist(labeled=False))
# print("dtree.todict(labeled=True) = %s" % dtree.todict(labeled=True))
# print("dtree.todict(labeled=False) = %s" % dtree.todict(labeled=False))

for index in range(len(tokens)):
    print("dtree.get_dependents(%d) = %s" % (index, dtree.get_dependents(index)))
    print("dtree.get_head(%d) = %s" % (index, dtree.get_head(index)))

ctree = treetk.dtree2ctree(dtree)
print("ctree.__str__() = %s" % ctree)
treetk.pretty_print(ctree)

rules = treetk.aggregate_production_rules(ctree)
print_list("production rules =", rules)

ctree.calc_spans()
spans = treetk.aggregate_spans(ctree)
print_list("spans =", spans)
mrg_spans = treetk.aggregate_composition_spans(ctree, binary=False)
print_list("composition spans =", mrg_spans)

subtree_strings = treetk.aggregate_subtrees(ctree, string=True)
print_list("subtrees =", subtree_strings)

print("tree2sexp(ctree) = %s" % treetk.tree2sexp(ctree))

if DRAW:
    treetk.draw(ctree)


