# -*- coding: utf-8 -*-

import sexp2tree

# Pattern 0
print("Pattern 0")
sexp = "( ( x0 x1 ) ( x2 ( x3 x4 ) ) )".split()
print("sexp =\n%s" % sexp)
tree = sexp2tree.sexp2tree(sexp, pattern=0)
print("tree (str) =\n%s" % tree)
print("tree (list) =\n%s" % tree.tolist())
tree.calc_ranges()
ranges = sexp2tree.aggregate_ranges(tree, acc=[])
print("ranges of constituents=\n%s" % ranges)
mrg_ranges = sexp2tree.aggregate_merging_ranges(tree, acc=[])
print("merging ranges of constituents=\n%s" % mrg_ranges)
print(tree.leaves())
print(tree.children[0].leaves())
print(tree.children[1].leaves())

# Patttern 1
print("Pattern 1")
sexp = "( A ( B x0 x1 ) ( C x2 ( D x3 x4 ) ) )".split()
print("sexp =\n%s" % sexp)
tree = sexp2tree.sexp2tree(sexp, pattern=1)
print("tree (str) =\n%s" % tree)
print("tree (list) =\n%s" % tree.tolist())
tree.calc_ranges()
ranges = sexp2tree.aggregate_ranges(tree, acc=[])
print("ranges of constituents=\n%s" % ranges)
mrg_ranges = sexp2tree.aggregate_merging_ranges(tree, acc=[])
print("merging ranges of constituents=\n%s" % mrg_ranges)
print(tree.leaves())
print(tree.children[0].leaves())
print(tree.children[1].leaves())

# Pattern 2
print("Pattern 2")
sexp = "( A ( B ( X0 x0 ) ( X1 x1 ) ) ( C ( X2 x2 ) ( D ( X3 x3 ) ( X4 x4 ) ) ) )".split()
print("sexp =\n%s" % sexp)
tree = sexp2tree.sexp2tree(sexp, pattern=2)
print("tree (str) =\n%s" % tree)
print("tree (list) =\n%s" % tree.tolist())
tree.calc_ranges()
ranges = sexp2tree.aggregate_ranges(tree, acc=[])
print("ranges of constituents=\n%s" % ranges)
mrg_ranges = sexp2tree.aggregate_merging_ranges(tree, acc=[])
print("merging ranges of constituents=\n%s" % mrg_ranges)
print(tree.leaves())
print(tree.children[0].leaves())
print(tree.children[1].leaves())


