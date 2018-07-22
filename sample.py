# -*- coding: utf-8 -*-

import treetk

DRAW=False

def print_list(msg, xs):
    print(msg)
    for x in xs:
        print("\t%s" % str(x))

###################################
print("\n############### Sample for labeled trees with POS tags ####################\n")
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

###################################
print("\n############### Sample for labeled trees without POS tags ####################\n")
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

###################################
print("\n############### Sample for unlabeled trees with POS tags ####################\n")
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

###################################
print("\n############### Sample for unlabeled trees without POS tags ####################\n")
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

###################################
print("\n############### Sample for unary or n-ary trees ####################\n")
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

###################################
print("\n############### Sample for dependency trees ####################\n")
tokens = ["<root>", "a", "boy", "saw", "a", "girl", "with", "a", "telescope"]
arcs = [(2, 1, "det"), (3, 2, "nsubj"), (3, 5, "dobj"), (5, 4, "det"), (3, 6, "prep"), (6, 8, "pobj"), (8, 7, "det"), (0, 3, "root")]
print("tokens = %s" % tokens)
print("arcs = %s" % arcs)
# dtree = treetk.produce_dependencytree(arcs=arcs) # this is allowable
dtree = treetk.produce_dependencytree(arcs=arcs, tokens=tokens)
print("dtree.__str__() = %s" % dtree)
print("dtree.tolist(labeled=True) = %s" % dtree.tolist(labeled=True))
print("dtree.tolist(labeled=False) = %s" % dtree.tolist(labeled=False))
print("dtree.head2dependents=%s" % dtree.head2dependents)
print("dtree.dependent2head=%s" % dtree.dependent2head)
for index in range(len(tokens)):
    print("dtree.get_dependents(%d) = %s" % (index, dtree.get_dependents(index)))
    print("dtree.get_head(%d) = %s" % (index, dtree.get_head(index)))

###################################
print("\n############### Sample for conversion of dependency tree -> constituency tree ####################\n")
print("dtree.__str__() = %s" % dtree)
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

###################################
print("\n############### Sample for conversion of constituency tree -> dependency tree ####################\n")
sexp = treetk.preprocess("(S (NP (DT a) (NN boy)) (VP (VP (VBD saw) (NP (DT a) (NN girl))) (PP (IN with) (NP (DT a) (NN telescope)))))".split())
ctree = treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=True)
print("ctree.__str__() = %s" % ctree)
def func_head_rule(node):
    # a simple example of the head-selection rules that specify a head subtree among the children nodes
    if node.label == "S" and node.children[0].label == "NP" and node.children[1].label == "VP":
        return 1 # the second child
    elif node.label == "NP" and node.children[0].label == "DT" and node.children[1].label == "NN":
        return 1
    elif node.label == "VP" and node.children[0].label == "VP" and node.children[1].label == "PP":
        return 0 # the first child
    elif node.label == "VP" and node.children[0].label == "VBD" and node.children[1].label == "NP":
        return 0
    elif node.label == "PP" and node.children[0].label == "IN" and node.children[1].label == "NP":
        return 0
    else:
        return 0
def func_label_rule(node, i, j):
    # a simple example of the rules that specify an arc label given parent and head/dependent nodes
    return "%s -> %s %s" % (node.label, node.children[i].label, node.children[j].label)
dtree = treetk.ctree2dtree(ctree, func_head_rule, func_label_rule)
print("dtree.__str__() = %s" % dtree)
