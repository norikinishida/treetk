# -*- coding: utf-8 -*-

import sexp2tree

# No labels for non-terminals and terminals
print("#############")
sexp = "( ( a cat ) ( bites ( a mouse ) ) )".split()
print("sexp = %s" % sexp)
tree = sexp2tree.sexp2tree(sexp, with_nonterminal_labels=False, with_terminal_labels=False)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())
tree.calc_ranges()
ranges = sexp2tree.aggregate_ranges(tree)
print("ranges of constituents = %s" % ranges)
mrg_ranges = sexp2tree.aggregate_merging_ranges(tree)
print("merging ranges of constituents = %s" % mrg_ranges)
sexp2tree.pretty_print(tree)
sexp2tree.draw(tree)

# Labels for non-terminals, but no labels for terminals
print("#############")
sexp = "( S ( NP a cat ) ( VP bites ( NP a mouse ) ) )".split()
print("sexp = %s" % sexp)
tree = sexp2tree.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())
tree.calc_ranges()
ranges = sexp2tree.aggregate_ranges(tree)
print("ranges of constituents = %s" % ranges)
mrg_ranges = sexp2tree.aggregate_merging_ranges(tree)
print("merging ranges of constituents = %s" % mrg_ranges)
sexp2tree.pretty_print(tree)
sexp2tree.draw(tree)

# Labels for both non-terminals and terminals
print("#############")
sexp = "( S ( NP ( DT a ) ( NN cat ) ) ( VP ( VBZ bites ) ( NP ( DT a ) ( NN mouse ) ) ) )".split()
print("sexp = %s" % sexp)
tree = sexp2tree.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=True)
print("tree.__str__() = %s" % tree)
print("tree.tolist() = %s" % tree.tolist())
print("tree.leaves() = %s" % tree.leaves())
print("tree.children[0].__str__() = %s" % tree.children[0])
print("tree.children[1].__str__() = %s" % tree.children[1])
print("tree.children[0].leaves() = %s" % tree.children[0].leaves())
print("tree.children[1].leaves() = %s" % tree.children[1].leaves())
tree.calc_ranges()
ranges = sexp2tree.aggregate_ranges(tree)
print("ranges of constituents = %s" % ranges)
mrg_ranges = sexp2tree.aggregate_merging_ranges(tree)
print("merging ranges of constituents = %s" % mrg_ranges)
sexp2tree.pretty_print(tree)
sexp2tree.draw(tree)

