import treetk
import utils

def test_ctree(sexp, with_nonterminal_labels, with_terminal_labels):
    # Preprocessing
    sexp = treetk.preprocess(sexp)
    print("S-expression:")
    print("\t%s" % sexp)

    # S-expression -> C-tree
    tree = treetk.sexp2tree(sexp, with_nonterminal_labels=with_nonterminal_labels, with_terminal_labels=with_terminal_labels)
    print("Constituent tree (with_nonterminal_labels=%s, with_terminal_labels=%s):" % (with_nonterminal_labels, with_terminal_labels))
    print("")
    treetk.pretty_print(tree)
    print("")

    # Traversing
    nodes = treetk.traverse(tree, order="pre-order", include_terminal=True, acc=None)
    print("Traversing (pre-order):")
    for node_i, node in enumerate(nodes):
        print("\t#%d" % (node_i+1))
        print("\tnode.is_terminal() = %s" % node.is_terminal())
        if node.is_terminal():
            if with_terminal_labels:
                print("\tnode.label = %s" % node.label)
            print("\tnode.token = %s" % node.token)
            print("\tnode.index = %s" % node.index)
        else:
            if with_nonterminal_labels:
                print("\tnode.label = %s" % node.label)
        print("\tstr(node) = %s" % str(node))
        print("\tnode.tolist() = %s" % node.tolist())
        print("\tnode.leaves() = %s" % node.leaves())
        if not node.is_terminal():
            for c_i in range(len(node.children)):
                print("\t\t#%d-%d" % (node_i+1, c_i+1))
                print("\t\tstr(node.children[%d]) = %s" % (c_i, str(node.children[c_i])))

    # Aggregation of production rules
    if with_nonterminal_labels:
        rules = treetk.aggregate_production_rules(tree, order="pre-order", include_terminal=with_terminal_labels)
        print("Aggregation of production rules (pre-order):")
        for rule in rules:
            print("\t%s" % str(rule))

    # Aggregation of spans
    tree.calc_spans() # NOTE

    spans = treetk.aggregate_spans(tree, include_terminal=False, order="pre-order")
    print("Aggregation of spans (w/o terminals, pre-order):")
    for span in spans:
        print("\t%s" % str(span))

    spans = treetk.aggregate_spans(tree, include_terminal=True, order="pre-order")
    print("Aggregation of spans (w/ terminals, pre-order):")
    for span in spans:
        print("\t%s" % str(span))

    mrg_spans = treetk.aggregate_composition_spans(tree, order="pre-order", binary=False)
    print("Aggregation of composition spans (pre-order):")
    for span in mrg_spans:
        print("\t%s" % str(span))

    # Aggregation of constituents
    constituents = treetk.aggregate_constituents(tree, order="pre-order")
    print("Aggregation of constituents (pre-order):")
    for constituent in constituents:
        print("\t%s" % str(constituent))

    # C-tree -> S-expression
    sexp = treetk.tree2sexp(tree)
    print("S-expression (reversed):")
    print("\t%s" % sexp)

print("\n############### Sample for labeled trees with POS tags ####################\n")
test_ctree("(S (NP (DT a) (NN cat)) (VP (VBZ bites) (NP (DT a) (NN mouse))))",
           with_nonterminal_labels=True,
           with_terminal_labels=True)

print("\n############### Sample for labeled trees without POS tags ####################\n")
test_ctree("(S (NP a cat) (VP bites (NP a mouse)))",
           with_nonterminal_labels=True,
           with_terminal_labels=False)

print("\n############### Sample for unlabeled trees with POS tags ####################\n")
test_ctree("(((DT a) (NN cat)) ((VBZ bites) ((DT a) (NN mouse))))",
           with_nonterminal_labels=False,
           with_terminal_labels=True)

print("\n############### Sample for unlabeled trees without POS tags ####################\n")
test_ctree("((a cat) (bites (a mouse)))",
           with_nonterminal_labels=False,
           with_terminal_labels=False)

print("\n############### Sample for unary or n-ary trees ####################\n")
test_ctree("(NP (NP (NP (N w0)) (NP (N w1))) (NP (N w2) (N w3) (N w4)))",
           with_nonterminal_labels=True,
           with_terminal_labels=True)

print("\n############### Sample for dependency trees ####################\n")
tokens = ["<root>", "a", "boy", "saw", "a", "girl", "with", "a", "telescope"]
arcs = [(2, 1, "det"), (3, 2, "nsubj"), (3, 5, "dobj"), (5, 4, "det"), (3, 6, "prep"), (6, 8, "pobj"), (8, 7, "det"), (0, 3, "<root>")]
print("tokens = %s" % tokens)
print("arcs = %s" % arcs)
dtree = treetk.arcs2dtree(arcs=arcs, tokens=tokens)
print("")
treetk.pretty_print_dtree(dtree)
print("")
print("str(dtree) = %s" % str(dtree))
print("dtree.tolist(labeled=True) = %s" % dtree.tolist(labeled=True))
print("dtree.tolist(labeled=False) = %s" % dtree.tolist(labeled=False))
print("dtree.head2dependents=%s" % dtree.head2dependents)
print("dtree.dependent2head=%s" % dtree.dependent2head)
for index in range(len(tokens)):
    print("\tToken %d" % index)
    print("\tdtree.get_head(%d) = %s" % (index, dtree.get_head(index)))
    print("\tdtree.get_dependents(%d) = %s" % (index, dtree.get_dependents(index)))

print("\n############### Sample for conversion from constituency tree to dependency tree ####################\n")
sexp = treetk.preprocess("(S (NP (DT a) (NN boy)) (VP (VP (VBD saw) (NP (DT a) (NN girl))) (PP (IN with) (NP (DT a) (NN telescope)))))")
ctree = treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=True)
treetk.pretty_print(ctree)
print("")

# Assign heads
def func_head_child_rule(node):
    """
    :type node: NonTerminal
    :rtype: int
    """
    # Please write your rules for specifying the head-child node among the child nodes
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
ctree.calc_heads(func_head_child_rule=func_head_child_rule) # NOTE

nodes = treetk.traverse(ctree, order="post-order", include_terminal=True, acc=None)
print("Heads (post-order):")
for node_i, node in enumerate(nodes):
    print("\t#%d" % (node_i+1))
    print("\tstr(node) = %s" % str(node))
    print("\tnode.head_child_index = %d" % node.head_child_index)
    print("\tnode.head_token_index = %d" % node.head_token_index)

def func_label_rule(node, i, j):
    """
    :type node: NonTerminal
    :type i: int
    :type j: int
    :rtype: str
    """
    # Please write a function that specifies the relation label (string) between the head (i.e., node.children[i]) and the dependent (i.e., node.children[j]).
    return node.label
    # return "%s,%s,%s" % (node.label, node.children[i].label, node.children[j].label)
dtree = treetk.ctree2dtree(ctree, func_label_rule=func_label_rule)
treetk.pretty_print_dtree(dtree)

print("\n############### Sample for conversion from dependency tree to constituency tree ####################\n")
tokens = ["<root>", "a", "boy", "saw", "a", "girl", "with", "a", "telescope"]
arcs = [(2, 1, "det"), (3, 2, "nsubj"), (3, 5, "dobj"), (5, 4, "det"), (3, 6, "prep"), (6, 8, "pobj"), (8, 7, "det"), (0, 3, "<root>")]
dtree = treetk.arcs2dtree(arcs=arcs, tokens=tokens)
treetk.pretty_print_dtree(dtree)
print("")

ctree = treetk.dtree2ctree(dtree)
treetk.pretty_print(ctree)

print("\n############### Sample for RST-DT constituency tree ####################\n")
sexp = utils.read_lines("./treetk/rstdt_example.labeled.nary.ctree", process=lambda line: line.split())[0]
print(" ".join(sexp))

ctree = treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False)
ctree = treetk.rstdt.postprocess(ctree)

ctree = treetk.rstdt.map_relations(ctree, mode="f2c")
treetk.pretty_print(ctree)
nodes = treetk.traverse(ctree, order="pre-order", include_terminal=False, acc=None)
for node in nodes:
    print(node.relation_label, node.nuclearity_label)

ctree = treetk.rstdt.map_relations(ctree, mode="c2a")
treetk.pretty_print(ctree)
nodes = treetk.traverse(ctree, order="pre-order", include_terminal=False, acc=None)
for node in nodes:
    print(node.relation_label, node.nuclearity_label)

