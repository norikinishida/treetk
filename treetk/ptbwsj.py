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
# Postprocessing

PUNCTUATIONS = ["-NONE-",
                ",", ".", "?", "!",
                "...",
                ":", ";",
                "``", "''",
                "--", "-",
                "-LRB-", "-RRB-",
                "-LCB-", "-RCB-",
                "(", ")", "{", "}",
                "$", "#"]

def lowercasing(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: NonTerminal or Terminal
    """
    if node.is_terminal():
        node.token = node.token.lower()
        return node

    # Recursive
    for c_i in range(len(node.children)):
        node.children[c_i] = lowercasing(node.children[c_i])

    return node

def remove_punctuations_and_empties(node, punctuations=None):
    """
    :type node: NonTerminal or Terminal
    :type punctuations: list of str
    :rtype: NonTerminal or Terminal
    """
    node = _remove_punctuations(node, punctuations=punctuations)
    node = _remove_empties(node)
    return node

def _remove_punctuations(node, punctuations=None):
    """
    :type node: NonTerminal or Terminal
    :type punctuations: list of str
    :rtype: NonTerminal or Terminal
    """
    if node.is_terminal():
        return node

    new_children = []
    for c_i in range(len(node.children)):
        # Remove child nodes that are terminals and punctuations.
        if punctuations is None:
            if node.children[c_i].is_terminal() and \
                    node.children[c_i].label in PUNCTUATIONS:
                continue
        else:
            if node.children[c_i].is_terminal() and \
                    node.children[c_i].label in punctuations:
                continue
        new_children.append(node.children[c_i])
    node.children = new_children

    # Recursive
    for c_i in range(len(node.children)):
        node.children[c_i] = _remove_punctuations(node.children[c_i], punctuations=punctuations)
    return node

def _remove_empties(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: NonTerminal or Terminal
    """
    if node.is_terminal():
        return node

    new_children = []
    for c_i in range(len(node.children)):
        # Remove non-terminal nodes without any child terminals.
        if _count_terminals(node.children[c_i]) > 0:
            new_children.append(node.children[c_i])
    node.children = new_children

    # Recursive
    for c_i in range(len(node.children)):
        node.children[c_i] = _remove_empties(node.children[c_i])

    return node

def _count_terminals(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: int
    """
    if node.is_terminal():
        return 1

    count = 0
    for c in node.children:
        count += _count_terminals(c)
    return count

def remove_function_tags(node):
    """
    :type node: NonTerminal or Terminal
    :rtype: NonTerminal or Terminal
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

def binarize(node, right_branching=True):
    """
    :type node: NonTerminal or Terminal
    :type right_branching: bool
    :rtype: NonTerminal or Terminal
    """
    if node.is_terminal():
        return node

    # Right/Left branching
    if len(node.children) > 2:
        if right_branching:
            node.children = _right_branching(node.children, parent_label=node.label)
        else:
            node.children = _left_branching(node.children, parent_label=node.label)

    # Recursive
    for c_i in range(len(node.children)):
        node.children[c_i] = binarize(node.children[c_i])

    return node

def _right_branching(nodes, parent_label):
    """
    :type nodes: list of NonTerminal/Terminal
    :type parent_label: str
    :rtype: [NonTerminal/Terminal, NonTerminal/Terminal]
    """
    if len(nodes) == 2:
        return nodes

    lhs = nodes[0] # The left-most child node is head

    # rhs = NonTerminal("+".join([n.label for n in nodes[1:]]))
    rhs = NonTerminal(parent_label if parent_label.endswith("^") else (parent_label + "^"))
    rhs.children = _right_branching(nodes[1:], parent_label=parent_label)

    return [lhs, rhs]

def _left_branching(nodes, parent_label):
    """
    :type nodes: list of NonTerminal/Terminal
    :type parent_label: str
    :rtype: [NonTerminal/Terminal, NonTerminal/Terminal]
    """
    if len(nodes) == 2:
        return nodes

    # lhs = NonTerminal("+".join([n.label for n in nodes[:-1]]))
    lhs = NonTerminal(parent_label if parent_label.endswith("^") else (parent_label + "^"))
    lhs.children = _left_branching(nodes[:-1], parent_label=parent_label)

    rhs = nodes[-1] # The right-most child node is head

    return [lhs, rhs]

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


