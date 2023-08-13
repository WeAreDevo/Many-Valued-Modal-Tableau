from dataclasses import dataclass
from syntax import AST_Node, parse_expression
from algebra import HeytingAlgebra, Poset
from collections import deque


class UniqueSymbolGenerator:
    def __init__(self):
        self.used_symbols = set()
        self.counter = 0

    def _int_to_symbol(self, i):
        # This is a simple conversion: integer to ASCII.
        # For a more complex unique encoding, modify this function.
        return chr(65 + i) if 0 <= i < 26 else f"S{i}"

    def get_symbol(self):
        while True:
            symbol = self._int_to_symbol(self.counter)
            self.counter += 1
            if symbol not in self.used_symbols:
                self.used_symbols.add(symbol)
                return symbol


gen = UniqueSymbolGenerator()


@dataclass(frozen=True)
class Signed_Formula:
    sign: str
    parsed_formula: AST_Node


class Tableau_Node:
    def __init__(
        self,
        world: str = None,
        relation: set[str] = None,
        parent=None,
        signed_form: Signed_Formula = None,
        children: list = None,
        isClosed: bool = False,
    ):
        self.world = world
        self.relation = relation
        self.parent = parent
        self.signed_form = signed_form
        if children:
            self.children = children
        else:
            self.children = []
        self.isClosed = isClosed


class Tableau:
    def __init__(self, root: Tableau_Node = None):
        self.root = root

    def isClosed(self):
        return self.root.isClosed


def checkClosed(node: Tableau_Node, H: HeytingAlgebra):
    signed_formula: Signed_Formula = node.signed_form
    sign = signed_formula.sign
    parsed_formula = signed_formula.parsed_formula

    if all(child.type == "VALUE" for child in parsed_formula.children):
        # TODO
        if sign == "T" and not H.poset.leq(
            parsed_formula.children[0].val, parsed_formula.children[1].val
        ):
            return True


def isAtomic(signed_formula: Signed_Formula):
    # TODO
    pass


def construct_tableau(signed_formula: str, H: HeytingAlgebra):
    root = Tableau_Node(
        world=gen.get_symbol(), relation=set(), signed_form=signed_formula
    )
    tableau = Tableau(root)
    q = deque()
    q.appendleft(root)

    while not tableau.isClosed() and not len(q) == 0:
        current_node: Tableau_Node = q.pop()
        if checkClosed(current_node, H):
            current_node.isClosed = True
        else:
            X: Signed_Formula = current_node.signed_form
            if isAtomic(X.parsed_formula):
                pass
    return


if __name__ == "__main__":
    expression = "a -> b"
    parsed_formula = parse_expression(expression)
    from PrettyPrint import PrettyPrintTree

    pt = PrettyPrintTree(lambda x: x.children, lambda x: x.val)
    pt(parsed_formula)
