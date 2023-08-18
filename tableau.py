from dataclasses import dataclass
from syntax import AST_Node, parse_expression
from algebra import TruthValue, HeytingAlgebra, Poset
from collections import deque


class UniqueSymbolGenerator:
    def __init__(self):
        self.used_symbols = set()
        self.counter = 0

    def _int_to_symbol(self, i):
        # This is a simple conversion: integer to ASCII.
        # For a more complex unique encoding, modify this function.
        return chr(65 + i) if 0 <= i < 26 else f"S{i}"

    def get_new_symbol(self):
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
        signed_formula: Signed_Formula = None,
        children: list = None,
        isClosed: bool = False,
    ):
        self.world = world
        self.relation = relation
        self.parent = parent
        self.signed_form = signed_formula
        if children:
            self.children = children
        else:
            self.children = []
        self.closed = isClosed


class Tableau:
    def __init__(self, root: Tableau_Node = None):
        self.root = root

    def isClosed(self):
        return self.root.closed


def isClosed(node: Tableau_Node, H: HeytingAlgebra):
    signed_formula: Signed_Formula = node.signed_form
    sign = signed_formula.sign
    parsed_formula = signed_formula.parsed_formula

    if all(isinstance(child.val, TruthValue) for child in parsed_formula.children):
        # p\bot_1
        if sign == "T" and not H.poset.leq(
            parsed_formula.children[0].val, parsed_formula.children[1].val
        ):
            return True
        # p\bot_2
        if sign == "F" and H.poset.leq(
            parsed_formula.children[0].val, parsed_formula.children[1].val
        ):
            return True
    # p\bot_3
    if sign == "F" and parsed_formula.children[0].val == H.bot:
        return True
    # p\bot_4
    if sign == "F" and parsed_formula.children[1].val == H.top:
        return True
    # p\bot_5 (TODO investigate if this is a derived rule i.e. not nescessary for completeness)
    if sign == "T" and isinstance(parsed_formula.children[0].val, TruthValue):
        curr = node.parent
        while curr != None:
            if (
                curr.world == node.world
                and curr.signed_formula.sign == "F"
                and isinstance(
                    curr.signed_formula.parsed_formula.children[0].val, TruthValue
                )
                and curr.signed_formula.parsed_formula.children[1]
                == parsed_formula.children[1]
            ):
                if H.poset.leq(
                    curr.signed_formula.parsed_formula.children[0].val,
                    parsed_formula.children[0].val,
                ):
                    return True
            curr = curr.parent
    # TODO: symmetrical case.
    if sign == "F" and isinstance(parsed_formula.children[0].val, TruthValue):
        curr = node.parent
        while curr != None:
            if (
                curr.world == node.world
                and curr.signed_formula.sign == "T"
                and isinstance(
                    curr.signed_formula.parsed_formula.children[0].val, TruthValue
                )
                and curr.signed_formula.parsed_formula.children[1]
                == parsed_formula.children[1]
            ):
                if H.poset.leq(
                    parsed_formula.children[0].val,
                    curr.signed_formula.parsed_formula.children[0].val,
                ):
                    return True
            curr = curr.parent


def isAtomic(parsed_formula: Signed_Formula):
    return (
        parsed_formula.children[0].type == "atom"
        and parsed_formula.children[1].type == "atom"
    )


def construct_tableau(signed_formula: str, H: HeytingAlgebra):
    root = Tableau_Node(
        world=gen.get_new_symbol(), relation=set(), signed_formula=signed_formula
    )
    tableau = Tableau(root)
    q = deque()
    q.appendleft(root)

    while not tableau.isClosed() and not len(q) == 0:
        current_node: Tableau_Node = q.pop()
        if current_node.closed:
            continue
        elif isClosed(current_node, H):
            current_node.closed = True
        else:
            X: Signed_Formula = current_node.signed_form
            if isAtomic(X.parsed_formula):
                pass
    return


if __name__ == "__main__":
    expression = "a -> b"
    signed_form = Signed_Formula("F", parse_expression(expression))

    bot = TruthValue("0")
    top = TruthValue("1")
    a = TruthValue("a")
    b = TruthValue("b")
    meetOp = {
        bot: {bot: bot, a: bot, b: bot, top: bot},
        a: {bot: bot, a: a, b: bot, top: a},
        b: {
            bot: bot,
            a: bot,
            b: b,
            top: b,
        },
        top: {
            bot: bot,
            a: a,
            b: b,
            top: top,
        },
    }
    ha = HeytingAlgebra({bot, a, b, top}, meetOp=meetOp)
    construct_tableau(signed_form, ha)
