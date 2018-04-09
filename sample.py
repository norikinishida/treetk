# -*- coding: utf-8 -*-

import sexp2tree

# Pattern 0: no labels for non-terminals and terminals
print("Pattern 0: no labels for non-terminals and terminals")
sexp = "( ( a cat ) ( bites ( a mouse ) ) )".split()
print("sexp =\n%s" % sexp)
tree = sexp2tree.sexp2tree(sexp, pattern=0)
print("tree (str) =\n%s" % tree)
print("tree (list) =\n%s" % tree.tolist())
print("tree.children[0].leaves() =\n%s" % tree.children[0].leaves())
print("tree.children[1].leaves() =\n%s" % tree.children[1].leaves())
tree.calc_ranges()
ranges = sexp2tree.aggregate_ranges(tree, with_nonterminal_labels=False)
print("ranges of constituents =\n%s" % ranges)
mrg_ranges = sexp2tree.aggregate_merging_ranges(tree, with_nonterminal_labels=False)
print("merging ranges of constituents =\n%s" % mrg_ranges)
sexp2tree.pretty_print(tree, pattern=0)

# Pattern 1: labels for non-terminals, but no labels for terminals
print("\nPattern 1: labels for non-terminals, but no labels for terminals")
sexp = "( S ( NP a cat ) ( VP bites ( NP a mouse ) ) )".split()
print("sexp =\n%s" % sexp)
tree = sexp2tree.sexp2tree(sexp, pattern=1)
print("tree (str) =\n%s" % tree)
print("tree (list) =\n%s" % tree.tolist())
print("tree.children[0].leaves() =\n%s" % tree.children[0].leaves())
print("tree.children[1].leaves() =\n%s" % tree.children[1].leaves())
tree.calc_ranges()
ranges = sexp2tree.aggregate_ranges(tree, with_nonterminal_labels=True)
print("ranges of constituents =\n%s" % ranges)
mrg_ranges = sexp2tree.aggregate_merging_ranges(tree, with_nonterminal_labels=True)
print("merging ranges of constituents =\n%s" % mrg_ranges)
sexp2tree.pretty_print(tree, pattern=1)

# Pattern 2: labels for non-terminals and terminals
print("\nPattern 2: labels for non-terminals and terminals")
sexp = "( S ( NP ( DT a ) ( NN cat ) ) ( VP ( VBZ bites ) ( NP ( DT a ) ( NN mouse ) ) ) )".split()
print("sexp =\n%s" % sexp)
tree = sexp2tree.sexp2tree(sexp, pattern=2)
print("tree (str) =\n%s" % tree)
print("tree (list) =\n%s" % tree.tolist())
print("tree.children[0].leaves() =\n%s" % tree.children[0].leaves())
print("tree.children[1].leaves() =\n%s" % tree.children[1].leaves())
tree.calc_ranges()
ranges = sexp2tree.aggregate_ranges(tree, with_nonterminal_labels=True)
print("ranges of constituents =\n%s" % ranges)
mrg_ranges = sexp2tree.aggregate_merging_ranges(tree, with_nonterminal_labels=True)
print("merging ranges of constituents =\n%s" % mrg_ranges)
sexp2tree.pretty_print(tree, pattern=2)

