# -*- coding: utf-8 -*-

import sexp2tree

DRAW=False

def print_list(msg, xs):
    print(msg)
    for x in xs:
        print("\t%s" % str(x))

# Labels for both non-terminals and terminals
print("#############")
sexp = sexp2tree.preprocess("(S (NP (DT a) (NN cat)) (VP (VBZ bites) (NP (DT a) (NN mouse))))")
print("sexp = %s" % sexp)

tree = sexp2tree.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=True)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].tolist() = %s" % tree.children[0].tolist())
print("tree.children[1].tolist() = %s" % tree.children[1].tolist())
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())

sexp2tree.pretty_print(tree)

rules = sexp2tree.aggregate_production_rules(tree)
print_list("production rules =", rules)

tree.calc_spans()
spans = sexp2tree.aggregate_spans(tree)
print_list("spans =", spans)
mrg_spans = sexp2tree.aggregate_composition_spans(tree)
print_list("composition spans =", mrg_spans)

subtree_strings = sexp2tree.aggregate_subtrees(tree, string=True)
print_list("subtrees =", subtree_strings)

print("tree2sexp(tree) = %s" % sexp2tree.tree2sexp(tree))

if DRAW:
    sexp2tree.draw(tree)

# Labels only for non-terminals, but no labels for terminals
print("#############")
sexp = sexp2tree.preprocess("(S (NP a cat) (VP bites (NP a mouse)))")
print("sexp = %s" % sexp)

tree = sexp2tree.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].tolist() = %s" % tree.children[0].tolist())
print("tree.children[1].tolist() = %s" % tree.children[1].tolist())
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())

sexp2tree.pretty_print(tree)

rules = sexp2tree.aggregate_production_rules(tree)
print_list("production rules =", rules)

tree.calc_spans()
spans = sexp2tree.aggregate_spans(tree)
print_list("spans =", spans)
mrg_spans = sexp2tree.aggregate_composition_spans(tree)
print_list("composition spans =", mrg_spans)

subtree_strings = sexp2tree.aggregate_subtrees(tree, string=True)
print_list("subtrees =", subtree_strings)

print("tree2sexp(tree) = %s" % sexp2tree.tree2sexp(tree))

if DRAW:
    sexp2tree.draw(tree)

# No labels for non-terminals, but only labels for terminals
print("#############")
sexp = sexp2tree.preprocess("(((DT a) (NN cat)) ((VBZ bites) ((DT a) (NN mouse))))")
print("sexp = %s" % sexp)

tree = sexp2tree.sexp2tree(sexp, with_nonterminal_labels=False, with_terminal_labels=True)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].tolist() = %s" % tree.children[0].tolist())
print("tree.children[1].tolist() = %s" % tree.children[1].tolist())
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())

sexp2tree.pretty_print(tree)

tree.calc_spans()
spans = sexp2tree.aggregate_spans(tree)
print_list("spans =", spans)
mrg_spans = sexp2tree.aggregate_composition_spans(tree)
print_list("composition spans =", mrg_spans)

subtree_strings = sexp2tree.aggregate_subtrees(tree, string=True)
print_list("subtrees =", subtree_strings)

print("tree2sexp(tree) = %s" % sexp2tree.tree2sexp(tree))

if DRAW:
    sexp2tree.draw(tree)

# No labels for non-terminals and terminals
print("#############")
sexp = sexp2tree.preprocess("((a cat) (bites (a mouse)))")
print("sexp = %s" % sexp)

tree = sexp2tree.sexp2tree(sexp, with_nonterminal_labels=False, with_terminal_labels=False)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].tolist() = %s" % tree.children[0].tolist())
print("tree.children[1].tolist() = %s" % tree.children[1].tolist())
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())

sexp2tree.pretty_print(tree)

tree.calc_spans()
spans = sexp2tree.aggregate_spans(tree)
print_list("spans =", spans)
mrg_spans = sexp2tree.aggregate_composition_spans(tree)
print_list("composition spans =", mrg_spans)

subtree_strings = sexp2tree.aggregate_subtrees(tree, string=True)
print_list("subtrees =", subtree_strings)

print("tree2sexp(tree) = %s" % sexp2tree.tree2sexp(tree))

if DRAW:
    sexp2tree.draw(tree)

# Can handle unary or n-ary trees
print("#############")
sexp = sexp2tree.preprocess("(NP (NP (NP (N w0)) (NP (N w1))) (NP (N w2) (N w3) (N w4)))")
print("sexp = %s" % sexp)

tree = sexp2tree.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=True)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].tolist() = %s" % tree.children[0].tolist())
print("tree.children[1].tolist() = %s" % tree.children[1].tolist())
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())

sexp2tree.pretty_print(tree)

rules = sexp2tree.aggregate_production_rules(tree)
print_list("production rules =", rules)

tree.calc_spans()
spans = sexp2tree.aggregate_spans(tree)
print_list("spans =", spans)
mrg_spans = sexp2tree.aggregate_composition_spans(tree, binary=False)
print_list("composition spans =", mrg_spans)

subtree_strings = sexp2tree.aggregate_subtrees(tree, string=True)
print_list("subtrees =", subtree_strings)

print("tree2sexp(tree) = %s" % sexp2tree.tree2sexp(tree))

if DRAW:
    sexp2tree.draw(tree)

