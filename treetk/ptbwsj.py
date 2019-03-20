from .ll import NonTerminal

############################
# IO

def read_sexps(path, LPAREN="(", RPAREN=")"):
    """
    :type path: str
    :rtype: list of list of str
    """
    sexps = []
    buf = []
    depth = 0
    for line in open(path):
        tokens = line.strip().replace("(", " ( ").replace(")", " ) ").split()
        if len(tokens) == 0:
            continue
        for token in tokens:
            if token == LPAREN:
                depth += 1
            elif token == RPAREN:
                depth -= 1
            buf.append(token)
            if depth == 0:
                # sexps.append(buf)
                sexps.append(buf[1:-1]) # Remove the outermost parens
                buf = []
    return sexps

############################
# Preprocessing

PUNCTUATIONS = ["``", "''", ":", ",", ".",
                "?", "!", ";", "...",
                "--", "-",
                "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                "(", ")", "{", "}",
                "$", "#"]

def lowercasing(node):
    """
    :type node: NonTerminal/Terminal
    :rtype: NonTerminal/Terminal
    """
    if node.is_terminal():
        node.token = node.token.lower()
        return node

    # Recursive
    for c_i in range(len(node.children)):
        node.children[c_i] = lowercasing(node.children[c_i])

    return node

def remove_function_tags(node):
    """
    :type node: NonTerminal/Terminal
    :rtype: NonTerminal/Terminal
    """
    if node.is_terminal():
        return node

    # Remove function tags from the label
    node.label = _remove_function_tags(node.label)

    # Recursive
    for c_i in range(len(node.children)):
        node.children[c_i] = remove_function_tags(node.children[c_i])

    return node

def _remove_function_tags(label):
    """
    :type label: str
    :rtype: str
    """
    if "-" in label and not label in ["-NONE-", "-LRB-", "-RRB-", "-LCB-", "-RCB-"]:
        lst = label.split("-")
        return lst[0]
    else:
        return label

def remove_empty_nodes(node):
    """
    :type node: NonTerminal/Terminal
    :rtype: NonTerminal/Terminal
    """
    node = _remove_nodes(node, removal_tags=["-NONE-"])
    node = _remove_invalid_nonterminals(node)
    return node

def remove_punctuations(node, punctuations=None):
    """
    :type node: NonTerminal/Terminal
    :type punctuations: list of str
    :rtype: NonTerminal/Terminal
    """
    if punctuations is None:
        punctuations = PUNCTUATIONS
    node = _remove_nodes(node, removal_tags=punctuations)
    node = _remove_invalid_nonterminals(node)
    return node

def _remove_nodes(node, removal_tags):
    """
    :type node: NonTerminal/Terminal
    :type removal_tags: list of str
    :rtype: NonTerminal/Terminal
    """
    if node.is_terminal():
        return node

    # Children of this node
    new_children = []
    for c_i in range(len(node.children)):
        # Remove (skip) child terminal nodes whose tags are in removal_tags.
        if node.children[c_i].is_terminal() and node.children[c_i].label in removal_tags:
            continue
        new_children.append(node.children[c_i])
    node.children = new_children

    # Recursive
    for c_i in range(len(node.children)):
        node.children[c_i] = _remove_nodes(node.children[c_i], removal_tags=removal_tags)
    return node

def _remove_invalid_nonterminals(node):
    """
    :type node: NonTerminal/Terminal
    :rtype: NonTerminal/Terminal
    """
    if node.is_terminal():
        return node

    # Children of this node
    new_children = []
    for c_i in range(len(node.children)):
        # Remove (skip) child nonterminal nodes without any child terminals.
        if _count_terminals(node.children[c_i]) == 0:
            continue
        new_children.append(node.children[c_i])
    node.children = new_children

    # Recursive
    for c_i in range(len(node.children)):
        node.children[c_i] = _remove_invalid_nonterminals(node.children[c_i])
    return node

def _count_terminals(node):
    """
    :type node: NonTerminal/Terminal
    :rtype: int
    """
    if node.is_terminal():
        return 1

    count = 0
    for c in node.children:
        count += _count_terminals(c)
    return count

def binarize(node, right_branching=True, special_empty_label=None):
    """
    :type node: NonTerminal/Terminal
    :type right_branching: bool
    :type special_empty_label: str
    :rtype: NonTerminal/Terminal
    """
    if node.is_terminal():
        return node

    # Right/Left branching
    if len(node.children) > 2:
        if right_branching:
            if special_empty_label is None:
                node.children = _right_branching(node.children,
                                                 inherit=True, parent_label=node.label)
            else:
                node.children = _right_branching(node.children,
                                                 inherit=False, special_empty_label=special_empty_label)
        else:
            if special_empty_label is None:
                node.children = _left_branching(node.children,
                                                inherit=True, parent_label=node.label)
            else:
                node.children = _left_branching(node.children,
                                                inherit=False, special_empty_label=special_empty_label)

    # Recursive
    for c_i in range(len(node.children)):
        node.children[c_i] = binarize(node.children[c_i],
                                      right_branching=right_branching,
                                      special_empty_label=special_empty_label)
    return node

def _right_branching(nodes, inherit, parent_label=None, special_empty_label=None):
    """
    :type nodes: list of NonTerminal/Terminal
    :type inherit: bool
    :type parent_label: str
    :type special_empty_label: str
    :rtype: [NonTerminal/Terminal, NonTerminal/Terminal]
    """
    if len(nodes) == 2:
        return nodes

    # Left node
    lhs = nodes[0] # The left-most child node is head
    # Right node
    if inherit:
        assert parent_label is not None
        assert special_empty_label is None
        rhs = NonTerminal(parent_label if parent_label.endswith("^") else (parent_label + "^"))
        rhs.children = _right_branching(nodes[1:], inherit=True, parent_label=parent_label)
    else:
        assert parent_label is None
        assert special_empty_label is not None
        rhs = NonTerminal(special_empty_label)
        rhs.children = _right_branching(nodes[1:], inherit=False, special_empty_label=special_empty_label)

    return [lhs, rhs]

def _left_branching(nodes, inherit, parent_label=None, special_empty_label=None):
    """
    :type nodes: list of NonTerminal/Terminal
    :type inherit: bool
    :type parent_label: str
    :type special_empty_label: str
    :rtype: [NonTerminal/Terminal, NonTerminal/Terminal]
    """
    if len(nodes) == 2:
        return nodes

    # Left node
    if inherit:
        assert parent_label is not None
        assert special_empty_label is None
        lhs = NonTerminal(parent_label if parent_label.endswith("^") else (parent_label + "^"))
        lhs.children = _left_branching(nodes[:-1], inherit=True, parent_label=parent_label)
    else:
        assert parent_label is None
        assert special_empty_label is not None
        lhs = NonTerminal(special_empty_label)
        lhs.children = _left_branching(nodes[:-1], inherit=False, special_empty_label=special_empty_label)
    # Right node
    rhs = nodes[-1] # The right-most child node is head

    return [lhs, rhs]

############################
# Postprocessing

def recover_narytree_by_removing_special_empty_labels(node, special_empty_label):
    """
    :type node: NonTerminal/Terminal
    :type special_empty_label: str
    :rtype: NonTerminal/Terminal
    """
    if node.is_terminal():
        return node

    # NOTE: Process in post order

    # Recursive
    for c_i in range(len(node.children)):
        node.children[c_i] = recover_narytree_by_removing_special_empty_labels(
                                node.children[c_i],
                                special_empty_label=special_empty_label)

    # Children of this node
    new_children = []
    for c_i in range(len(node.children)):
        if node.children[c_i].is_terminal():
            new_children.append(node.children[c_i])
        elif node.children[c_i].label != special_empty_label:
            new_children.append(node.children[c_i])
        else:
            new_children.extend(node.children[c_i].children)
    node.children = new_children

    return node

############################
# Others

# def add_dummy_node(node):
#     """
#     :type node: NonTerminal or Terminal
#     :rtype: NonTerminal or Terminal
#     """
#     if node.is_terminal():
#         return node
#     if len(node.children) == 1:
#         new_node = Terminal(label="<DUMMY>", token="___")
#         node.add_child(new_node)
#     for c_i in range(len(node.children)):
#         node.children[c_i] = add_dummy_node(node.children[c_i])
#     return node


