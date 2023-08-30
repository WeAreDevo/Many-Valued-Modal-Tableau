from dataclasses import dataclass
import functools
from syntax import AST_Node, parse_expression
from algebra import TruthValue, HeytingAlgebra, Poset
from collections import deque
import copy
from PrettyPrint import PrettyPrintTree
import re


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
    parse_tree: AST_Node

    def __str__(self) -> str:
        return f"{self.sign} {str(self.parse_tree)}"


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
        self.signed_formula = signed_formula
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
    signed_formula: Signed_Formula = node.signed_formula
    sign = signed_formula.sign
    parse_tree = signed_formula.parse_tree

    if all(
        isinstance(child.val, TruthValue) for child in parse_tree.proper_subformulas
    ):
        # p\bot_1
        if sign == "T" and not H.poset.leq(
            parse_tree.proper_subformulas[0].val,
            parse_tree.proper_subformulas[1].val,
        ):
            return True
        # p\bot_2
        if sign == "F" and H.poset.leq(
            parse_tree.proper_subformulas[0].val,
            parse_tree.proper_subformulas[1].val,
        ):
            return True
    # p\bot_3
    if sign == "F" and parse_tree.proper_subformulas[0].val == H.bot:
        return True
    # p\bot_4
    if sign == "F" and parse_tree.proper_subformulas[1].val == H.top:
        return True
    # p\bot_5 (TODO investigate if this is a derived rule i.e. not nescessary for completeness)
    if sign == "T" and isinstance(parse_tree.proper_subformulas[0].val, TruthValue):
        curr = node.parent
        while curr != None:
            if (
                curr.world == node.world
                and curr.signed_formula.sign == "F"
                and isinstance(
                    curr.signed_formula.parse_tree.proper_subformulas[0].val, TruthValue
                )
                and curr.signed_formula.parse_tree.proper_subformulas[1]
                == parse_tree.proper_subformulas[1]
            ):
                if H.poset.leq(
                    curr.signed_formula.parse_tree.proper_subformulas[0].val,
                    parse_tree.proper_subformulas[0].val,
                ):
                    return True
            curr = curr.parent
    # symmetrical case.
    if sign == "F" and isinstance(parse_tree.proper_subformulas[0].val, TruthValue):
        curr = node.parent
        while curr != None:
            if (
                curr.world == node.world
                and curr.signed_formula.sign == "T"
                and isinstance(
                    curr.signed_formula.parse_tree.proper_subformulas[0].val, TruthValue
                )
                and curr.signed_formula.parse_tree.proper_subformulas[1]
                == parse_tree.proper_subformulas[1]
            ):
                if H.poset.leq(
                    parse_tree.proper_subformulas[0].val,
                    curr.signed_formula.parse_tree.proper_subformulas[0].val,
                ):
                    return True
            curr = curr.parent

    # better \bot5?
    if sign == "T" and isinstance(parse_tree.proper_subformulas[0].val, TruthValue):
        curr = node.parent
        while curr != None:
            if (
                curr.world == node.world
                and curr.signed_formula.sign == "T"
                and isinstance(
                    curr.signed_formula.parse_tree.proper_subformulas[1].val, TruthValue
                )
                and curr.signed_formula.parse_tree.proper_subformulas[0]
                == parse_tree.proper_subformulas[1]
            ):
                if not H.poset.leq(
                    parse_tree.proper_subformulas[0].val,
                    curr.signed_formula.parse_tree.proper_subformulas[1].val,
                ):
                    return True
            curr = curr.parent
    # symmetrical case.
    if sign == "T" and isinstance(parse_tree.proper_subformulas[1].val, TruthValue):
        curr = node.parent
        while curr != None:
            if (
                curr.world == node.world
                and curr.signed_formula.sign == "T"
                and isinstance(
                    curr.signed_formula.parse_tree.proper_subformulas[0].val, TruthValue
                )
                and curr.signed_formula.parse_tree.proper_subformulas[1]
                == parse_tree.proper_subformulas[0]
            ):
                if not H.poset.leq(
                    curr.signed_formula.parse_tree.proper_subformulas[0].val,
                    parse_tree.proper_subformulas[1].val,
                ):
                    return True
            curr = curr.parent


def isAtomic(parse_tree: AST_Node):
    return (
        parse_tree.proper_subformulas[0].type == "atom"
        and parse_tree.proper_subformulas[1].type == "atom"
    )


def forkOpenBranches(node: Tableau_Node, children: list[Tableau_Node], q: deque):
    # DFS
    if node.closed:
        return
    if not node.children:
        children_copy = copy.deepcopy(children)
        for c in children_copy:
            c.parent = node
            curr = [c]
            while curr:
                q.appendleft(curr[0])
                curr = curr[0].children
        node.children = children_copy
        return
    for child in node.children:
        forkOpenBranches(child, children, q)


def ApplyFleq(curr: Tableau_Node, q: deque[Tableau_Node], H: HeytingAlgebra):
    signed_form: Signed_Formula = curr.signed_formula
    X = {
        u
        for u in H.elements
        if not H.poset.leq(u, signed_form.parse_tree.proper_subformulas[1].val)
    }
    new_nodes = []
    for u in H.poset.minimals(X):
        proper_subformulas = [
            AST_Node("atom", u),
            copy.deepcopy(signed_form.parse_tree.proper_subformulas[0]),
        ]
        new_form = AST_Node(
            type=signed_form.parse_tree.type,
            val=signed_form.parse_tree.val,
            proper_subformulas=proper_subformulas,
        )
        new_signed_formula = Signed_Formula(sign="T", parse_tree=new_form)
        n = Tableau_Node(
            world=curr.world,
            relation=copy.copy(curr.relation),
            signed_formula=new_signed_formula,
        )
        new_nodes.append(n)
    forkOpenBranches(curr, new_nodes, q)


def ApplyTleq(curr: Tableau_Node, q: deque[Tableau_Node], H: HeytingAlgebra):
    signed_form: Signed_Formula = curr.signed_formula
    if signed_form.parse_tree.proper_subformulas[1].val == H.top:  # Side condition
        return
    X = {
        u
        for u in H.elements
        if not H.poset.leq(u, signed_form.parse_tree.proper_subformulas[1].val)
    }
    new_nodes = []
    for u in H.poset.minimals(X):
        proper_subformulas = [
            AST_Node("atom", u),
            copy.deepcopy(signed_form.parse_tree.proper_subformulas[0]),
        ]
        new_form = AST_Node(
            type=signed_form.parse_tree.type,
            val=signed_form.parse_tree.val,
            proper_subformulas=proper_subformulas,
        )
        new_signed_formula = Signed_Formula(sign="F", parse_tree=new_form)
        n = Tableau_Node(
            world=curr.world,
            relation=copy.copy(curr.relation),
            signed_formula=new_signed_formula,
        )
        new_nodes.append(n)
    for i in range(len(new_nodes) - 1):
        new_nodes[i].children = [new_nodes[i + 1]]
    forkOpenBranches(curr, [new_nodes[0]], q)


def ApplyTgeq(curr: Tableau_Node, q: deque[Tableau_Node], H: HeytingAlgebra):
    signed_form: Signed_Formula = curr.signed_formula
    if signed_form.parse_tree.proper_subformulas[0].val == H.bot:  # Side condition
        return
    X = {
        t
        for t in H.elements
        if not H.poset.leq(signed_form.parse_tree.proper_subformulas[0].val, t)
    }
    new_nodes = []
    for t in H.poset.maximals(X):
        proper_subformulas = [
            copy.deepcopy(signed_form.parse_tree.proper_subformulas[1]),
            AST_Node("atom", t),
        ]
        new_form = AST_Node(
            type=signed_form.parse_tree.type,
            val=signed_form.parse_tree.val,
            proper_subformulas=proper_subformulas,
        )
        new_signed_formula = Signed_Formula(sign="F", parse_tree=new_form)
        n = Tableau_Node(
            world=curr.world,
            relation=copy.copy(curr.relation),
            signed_formula=new_signed_formula,
        )
        new_nodes.append(n)
    for i in range(len(new_nodes) - 1):
        new_nodes[i].children = [new_nodes[i + 1]]
    forkOpenBranches(curr, [new_nodes[0]], q)


def ApplyFgeq(curr: Tableau_Node, q: deque[Tableau_Node], H: HeytingAlgebra):
    signed_form: Signed_Formula = curr.signed_formula
    X = {
        u
        for u in H.elements
        if not H.poset.leq(signed_form.parse_tree.proper_subformulas[0].val, u)
    }
    new_nodes = []
    for u in H.poset.maximals(X):
        proper_subformulas = [
            copy.deepcopy(signed_form.parse_tree.proper_subformulas[1]),
            AST_Node("atom", u),
        ]
        new_form = AST_Node(
            type=signed_form.parse_tree.type,
            val=signed_form.parse_tree.val,
            proper_subformulas=proper_subformulas,
        )
        new_signed_formula = Signed_Formula(sign="T", parse_tree=new_form)
        n = Tableau_Node(
            world=curr.world,
            relation=copy.copy(curr.relation),
            signed_formula=new_signed_formula,
        )
        new_nodes.append(n)
    forkOpenBranches(curr, new_nodes, q)


def update_closed(node: Tableau_Node, H: HeytingAlgebra):
    if isClosed(node, H):
        node.closed = True
        return
    for child in node.children:
        update_closed(child, H)
    if node.children and all(c.closed for c in node.children):
        node.closed = True
        return


def get_open_branches(node: Tableau_Node):
    branches = []

    def dfs(n: Tableau_Node, current_path: deque[Tableau_Node]):
        current_path.append(n)
        if n.closed:
            return

        if not n.children:
            branches.append(current_path.copy())
            return

        for c in n.children:
            dfs(c, current_path)
            current_path.pop()

    dfs(node, deque())
    ancestors = deque()
    curr = node.parent
    while curr:
        ancestors.appendleft(curr)
        curr = curr.parent
    if ancestors:
        for b in branches:
            b.extendleft(list(ancestors))
    return branches


def worlds(S: list[Tableau_Node]):
    return functools.reduce(lambda x, y: x.union(y), [{n.world} for n in S])


def cons(S: list[Tableau_Node]):
    return functools.reduce(lambda x, y: x.union(y), [n.relation for n in S])


def ApplyTbox(current_node, q, H):
    X: Signed_Formula = current_node.signed_formula
    phi: AST_Node = copy.deepcopy(
        X.parse_tree.proper_subformulas[1].proper_subformulas[0]
    )
    pattern = f"{current_node.world}#(.*)#(.*)"
    for S in get_open_branches(current_node):
        for t, world in [
            (m.group(1), m.group(2)) for r in cons(S) if (m := re.match(pattern, r))
        ]:
            proper_subformulas = [
                AST_Node(
                    type="atom",
                    val=H.meet(
                        X.parse_tree.proper_subformulas[0].val,
                        TruthValue(t),
                    ),
                ),
                phi,
            ]
            new_signed_formula = Signed_Formula(
                sign="T",
                parse_tree=AST_Node(
                    type=X.parse_tree.type,
                    val=X.parse_tree.val,
                    proper_subformulas=proper_subformulas,
                ),
            )
            n = Tableau_Node(
                world=world,
                relation=current_node.relation,
                signed_formula=new_signed_formula,
                parent=S[-1],
            )
            S[-1].children = [n]
            S.append(n)
            q.appendleft(n)


def ApplyTdiamond(current_node, q, H):
    X: Signed_Formula = current_node.signed_formula
    phi: AST_Node = copy.deepcopy(
        X.parse_tree.proper_subformulas[0].proper_subformulas[0]
    )
    pattern = f"{current_node.world}#(.*)#(.*)"
    for S in get_open_branches(current_node):
        for t, world in [
            (m.group(1), m.group(2)) for r in cons(S) if (m := re.match(pattern, r))
        ]:
            proper_subformulas = [
                phi,
                AST_Node(
                    type="atom",
                    val=H.implies(
                        TruthValue(t),
                        X.parse_tree.proper_subformulas[1].val,
                    ),
                ),
            ]
            new_signed_formula = Signed_Formula(
                sign="T",
                parse_tree=AST_Node(
                    type=X.parse_tree.type,
                    val=X.parse_tree.val,
                    proper_subformulas=proper_subformulas,
                ),
            )
            n = Tableau_Node(
                world=world,
                relation=current_node.relation,
                signed_formula=new_signed_formula,
                parent=S[-1],
            )
            S[-1].children = [n]
            S.append(n)
            q.appendleft(n)


def reactivate(current_node, q, H):
    pred = current_node.parent
    while pred:
        if pred.world == current_node.world and pred.signed_formula.sign == "T":
            if (
                isinstance(
                    pred.signed_formula.parse_tree.proper_subformulas[0].val,
                    TruthValue,
                )
                and pred.signed_formula.parse_tree.proper_subformulas[1].val == "[]"
            ):
                ApplyTbox(pred, q, H)
            elif (
                isinstance(
                    pred.signed_formula.parse_tree.proper_subformulas[1].val,
                    TruthValue,
                )
                and pred.signed_formula.parse_tree.proper_subformulas[0].val == "<>"
            ):
                ApplyTdiamond(pred, q, H)
        pred = pred.parent


def construct_tableau(
    input_signed_formula: Signed_Formula, H: HeytingAlgebra, print=True
):
    root = Tableau_Node(
        world=gen.get_new_symbol(), relation=set(), signed_formula=input_signed_formula
    )
    tableau = Tableau(root)
    q = deque()
    q.appendleft(root)

    while not len(q) == 0:
        current_node: Tableau_Node = q.pop()
        if current_node.closed:
            continue
        # elif isClosed(current_node, H):  # only at end?
        #     current_node.closed = True
        else:
            X: Signed_Formula = current_node.signed_formula

            # ATOMIC
            if isAtomic(X.parse_tree):
                # Check if reversal rule sould be applied
                if X.sign == "F":
                    if not isinstance(
                        X.parse_tree.proper_subformulas[0].val,
                        TruthValue,
                    ):
                        ApplyFleq(current_node, q, H)
                    elif not isinstance(
                        X.parse_tree.proper_subformulas[1].val,
                        TruthValue,
                    ):
                        ApplyFgeq(current_node, q, H)
                continue

            # T&
            # Check if reversal should be applied first
            elif (
                X.sign == "F"
                and isinstance(X.parse_tree.proper_subformulas[1].val, TruthValue)
                and X.parse_tree.proper_subformulas[0].val == "&"
            ):
                ApplyFleq(current_node, q, H)
                continue
            elif (
                X.sign == "T"
                and isinstance(X.parse_tree.proper_subformulas[0].val, TruthValue)
                and X.parse_tree.proper_subformulas[1].val == "&"
                and not X.parse_tree.proper_subformulas[0].val
                == H.bot  # Side condition
            ):
                phi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[1].proper_subformulas[0]
                )
                psi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[1].proper_subformulas[1]
                )
                proper_subformulas = [
                    X.parse_tree.proper_subformulas[0],
                    phi,
                ]
                new_form = AST_Node(
                    type=X.parse_tree.type,
                    val=X.parse_tree.val,
                    proper_subformulas=proper_subformulas,
                )
                new_signed_formula = Signed_Formula(sign="T", parse_tree=new_form)
                nl = Tableau_Node(
                    world=current_node.world,
                    relation=current_node.relation,
                    parent=None,
                    signed_formula=new_signed_formula,
                )

                proper_subformulas = [
                    X.parse_tree.proper_subformulas[0],
                    psi,
                ]
                new_form = AST_Node(
                    type=X.parse_tree.type,
                    val=X.parse_tree.val,
                    proper_subformulas=proper_subformulas,
                )
                new_signed_formula = Signed_Formula(sign="T", parse_tree=new_form)
                nr = Tableau_Node(
                    world=current_node.world,
                    relation=current_node.relation,
                    parent=nl,
                    signed_formula=new_signed_formula,
                )
                nl.children = [nr]
                forkOpenBranches(current_node, [nl], q)

            # F&
            # Check if reversal should be applied first
            elif (
                X.sign == "T"
                and isinstance(X.parse_tree.proper_subformulas[1].val, TruthValue)
                and X.parse_tree.proper_subformulas[0].val == "&"
            ):
                ApplyTleq(current_node, q, H)
                continue
            elif (
                X.sign == "F"
                and isinstance(X.parse_tree.proper_subformulas[0].val, TruthValue)
                and X.parse_tree.proper_subformulas[1].val == "&"
            ):
                phi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[1].proper_subformulas[0]
                )
                psi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[1].proper_subformulas[1]
                )
                proper_subformulas = [
                    X.parse_tree.proper_subformulas[0],
                    phi,
                ]
                new_form = AST_Node(
                    type=X.parse_tree.type,
                    val=X.parse_tree.val,
                    proper_subformulas=proper_subformulas,
                )
                new_signed_formula = Signed_Formula(sign="F", parse_tree=new_form)
                nl = Tableau_Node(
                    world=current_node.world,
                    relation=current_node.relation,
                    signed_formula=new_signed_formula,
                )

                proper_subformulas = [
                    X.parse_tree.proper_subformulas[0],
                    psi,
                ]
                new_form = AST_Node(
                    type=X.parse_tree.type,
                    val=X.parse_tree.val,
                    proper_subformulas=proper_subformulas,
                )
                new_signed_formula = Signed_Formula(sign="F", parse_tree=new_form)
                nr = Tableau_Node(
                    world=current_node.world,
                    relation=current_node.relation,
                    signed_formula=new_signed_formula,
                )
                forkOpenBranches(current_node, [nl, nr], q)

            # T|
            # Check if reversal should be applied first
            elif (
                X.sign == "F"
                and isinstance(X.parse_tree.proper_subformulas[0].val, TruthValue)
                and X.parse_tree.proper_subformulas[1].val == "|"
            ):
                ApplyFgeq(current_node, q, H)
                continue
            elif (
                X.sign == "T"
                and isinstance(X.parse_tree.proper_subformulas[1].val, TruthValue)
                and X.parse_tree.proper_subformulas[0].val == "|"
                and not X.parse_tree.proper_subformulas[1].val
                == H.top  # Side Condition
            ):
                phi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[0].proper_subformulas[0]
                )
                psi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[0].proper_subformulas[1]
                )
                proper_subformulas = [
                    phi,
                    X.parse_tree.proper_subformulas[1],
                ]
                new_form = AST_Node(
                    type=X.parse_tree.type,
                    val=X.parse_tree.val,
                    proper_subformulas=proper_subformulas,
                )
                new_signed_formula = Signed_Formula(sign="T", parse_tree=new_form)
                nl = Tableau_Node(
                    world=current_node.world,
                    relation=current_node.relation,
                    parent=None,
                    signed_formula=new_signed_formula,
                )

                proper_subformulas = [
                    psi,
                    X.parse_tree.proper_subformulas[1],
                ]
                new_form = AST_Node(
                    type=X.parse_tree.type,
                    val=X.parse_tree.val,
                    proper_subformulas=proper_subformulas,
                )
                new_signed_formula = Signed_Formula(sign="T", parse_tree=new_form)
                nr = Tableau_Node(
                    world=current_node.world,
                    relation=current_node.relation,
                    parent=nl,
                    signed_formula=new_signed_formula,
                )
                nl.children = [nr]
                forkOpenBranches(current_node, [nl], q)

            # F|
            # Check if reversal should be applied first
            elif (
                X.sign == "T"
                and isinstance(X.parse_tree.proper_subformulas[0].val, TruthValue)
                and X.parse_tree.proper_subformulas[1].val == "|"
            ):
                ApplyTgeq(current_node, q, H)
                continue
            elif (
                X.sign == "F"
                and isinstance(X.parse_tree.proper_subformulas[1].val, TruthValue)
                and X.parse_tree.proper_subformulas[0].val == "|"
            ):
                phi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[0].proper_subformulas[0]
                )
                psi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[0].proper_subformulas[1]
                )
                proper_subformulas = [
                    phi,
                    X.parse_tree.proper_subformulas[1],
                ]
                new_form = AST_Node(
                    type=X.parse_tree.type,
                    val=X.parse_tree.val,
                    proper_subformulas=proper_subformulas,
                )
                new_signed_formula = Signed_Formula(sign="F", parse_tree=new_form)
                nl = Tableau_Node(
                    world=current_node.world,
                    relation=current_node.relation,
                    signed_formula=new_signed_formula,
                )

                proper_subformulas = [
                    psi,
                    X.parse_tree.proper_subformulas[1],
                ]
                new_form = AST_Node(
                    type=X.parse_tree.type,
                    val=X.parse_tree.val,
                    proper_subformulas=proper_subformulas,
                )
                new_signed_formula = Signed_Formula(sign="F", parse_tree=new_form)
                nr = Tableau_Node(
                    world=current_node.world,
                    relation=current_node.relation,
                    signed_formula=new_signed_formula,
                )
                forkOpenBranches(current_node, [nl, nr], q)

            # F->
            # Check if reversal should be applied first
            elif (
                X.sign == "T"
                and isinstance(X.parse_tree.proper_subformulas[1].val, TruthValue)
                and X.parse_tree.proper_subformulas[0].val == "->"
            ):
                ApplyTleq(current_node, q, H)
                continue
            elif (
                X.sign == "F"
                and isinstance(X.parse_tree.proper_subformulas[0].val, TruthValue)
                and X.parse_tree.proper_subformulas[1].val == "->"
            ):
                phi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[1].proper_subformulas[0]
                )
                psi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[1].proper_subformulas[1]
                )
                elems = {
                    t
                    for t in H.elements
                    if not t == H.bot
                    and H.poset.leq(t, X.parse_tree.proper_subformulas[0].val)
                }
                children = []
                for t in H.poset.minimals(elems):
                    proper_subformulas1 = [
                        AST_Node("atom", t),
                        phi,
                    ]
                    proper_subformulas2 = [
                        AST_Node("atom", t),
                        psi,
                    ]
                    new_signed_formula1 = Signed_Formula(
                        sign="T",
                        parse_tree=AST_Node(
                            type=X.parse_tree.type,
                            val=X.parse_tree.val,
                            proper_subformulas=proper_subformulas1,
                        ),
                    )
                    new_signed_formula2 = Signed_Formula(
                        sign="F",
                        parse_tree=AST_Node(
                            type=X.parse_tree.type,
                            val=X.parse_tree.val,
                            proper_subformulas=proper_subformulas2,
                        ),
                    )
                    n1 = Tableau_Node(
                        world=current_node.world,
                        relation=current_node.relation,
                        signed_formula=new_signed_formula1,
                    )
                    n2 = Tableau_Node(
                        world=current_node.world,
                        relation=current_node.relation,
                        signed_formula=new_signed_formula2,
                    )
                    n1.children = [n2]
                    n2.parent = n1
                    children.append(n1)
                forkOpenBranches(current_node, children, q)

            # T->
            # Check if reversal should be applied first
            elif (
                X.sign == "F"
                and isinstance(X.parse_tree.proper_subformulas[1].val, TruthValue)
                and X.parse_tree.proper_subformulas[0].val == "->"
            ):
                ApplyFleq(current_node, q, H)
                continue
            elif (
                X.sign == "T"
                and isinstance(X.parse_tree.proper_subformulas[0].val, TruthValue)
                and X.parse_tree.proper_subformulas[1].val == "->"
                and not X.parse_tree.proper_subformulas[0].val
                == H.bot  # Side Condition
            ):
                phi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[1].proper_subformulas[0]
                )
                psi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[1].proper_subformulas[1]
                )
                elems = {
                    t
                    for t in H.elements
                    if not t == H.bot
                    and H.poset.leq(t, X.parse_tree.proper_subformulas[0].val)
                }
                children = []
                for t in elems:
                    proper_subformulas1 = [
                        AST_Node("atom", t),
                        phi,
                    ]
                    proper_subformulas2 = [
                        AST_Node("atom", t),
                        psi,
                    ]
                    new_signed_formula1 = Signed_Formula(
                        sign="F",
                        parse_tree=AST_Node(
                            type=X.parse_tree.type,
                            val=X.parse_tree.val,
                            proper_subformulas=proper_subformulas1,
                        ),
                    )
                    new_signed_formula2 = Signed_Formula(
                        sign="T",
                        parse_tree=AST_Node(
                            type=X.parse_tree.type,
                            val=X.parse_tree.val,
                            proper_subformulas=proper_subformulas2,
                        ),
                    )
                    n1 = Tableau_Node(
                        world=current_node.world,
                        relation=current_node.relation,
                        signed_formula=new_signed_formula1,
                    )
                    n2 = Tableau_Node(
                        world=current_node.world,
                        relation=current_node.relation,
                        signed_formula=new_signed_formula2,
                    )
                    children.extend([n1, n2])
                forkOpenBranches(current_node, children, q)

            # T[]
            # Check if reversal should be applied first
            elif (
                X.sign == "F"
                and isinstance(X.parse_tree.proper_subformulas[1].val, TruthValue)
                and X.parse_tree.proper_subformulas[0].val == "[]"
            ):
                ApplyFleq(current_node, q, H)
                continue
            elif (
                X.sign == "T"
                and isinstance(X.parse_tree.proper_subformulas[0].val, TruthValue)
                and X.parse_tree.proper_subformulas[1].val == "[]"
                # TODO: Side condition fine to add?
            ):
                ApplyTbox(current_node, q, H)

            # T<>
            # Check if reversal should be applied first
            elif (
                X.sign == "F"
                and isinstance(X.parse_tree.proper_subformulas[0].val, TruthValue)
                and X.parse_tree.proper_subformulas[1].val == "<>"
            ):
                ApplyFgeq(current_node, q, H)
                continue
            elif (
                X.sign == "T"
                and isinstance(X.parse_tree.proper_subformulas[1].val, TruthValue)
                and X.parse_tree.proper_subformulas[0].val == "<>"
            ):
                ApplyTdiamond(current_node, q, H)

            # F[]
            # Check if reversal should be applied first
            elif (
                X.sign == "T"
                and isinstance(X.parse_tree.proper_subformulas[1].val, TruthValue)
                and X.parse_tree.proper_subformulas[0].val == "[]"
            ):
                ApplyTleq(current_node, q, H)
                continue
            elif (
                X.sign == "F"
                and isinstance(X.parse_tree.proper_subformulas[0].val, TruthValue)
                and X.parse_tree.proper_subformulas[1].val == "[]"
            ):
                phi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[1].proper_subformulas[0]
                )
                for S in get_open_branches(current_node):
                    for t in H.elements:
                        if H.meet(t, X.parse_tree.proper_subformulas[0].val) == H.bot:
                            continue

                        proper_subformulas = [
                            AST_Node(
                                type="atom",
                                val=H.meet(
                                    X.parse_tree.proper_subformulas[0].val,
                                    t,
                                ),
                            ),
                            phi,
                        ]
                        new_signed_formula = Signed_Formula(
                            sign="F",
                            parse_tree=AST_Node(
                                type=X.parse_tree.type,
                                val=X.parse_tree.val,
                                proper_subformulas=proper_subformulas,
                            ),
                        )
                        new_world = gen.get_new_symbol()
                        new_relation = current_node.relation.union(
                            {f"{current_node.world}#{t.value}#{new_world}"}
                        )
                        n = Tableau_Node(
                            world=new_world,
                            relation=new_relation,
                            signed_formula=new_signed_formula,
                            parent=S[-1],
                        )
                        S[-1].children.append(n)
                        q.append(n)
                reactivate(current_node, q, H)

            # F<>
            # Check if reversal should be applied first
            elif (
                X.sign == "T"
                and isinstance(X.parse_tree.proper_subformulas[0].val, TruthValue)
                and X.parse_tree.proper_subformulas[1].val == "<>"
            ):
                ApplyTgeq(current_node, q, H)
                continue
            elif (
                X.sign == "F"
                and isinstance(X.parse_tree.proper_subformulas[1].val, TruthValue)
                and X.parse_tree.proper_subformulas[0].val == "<>"
            ):
                phi: AST_Node = copy.deepcopy(
                    X.parse_tree.proper_subformulas[0].proper_subformulas[0]
                )
                for S in get_open_branches(current_node):
                    for t in H.elements:
                        if (
                            H.implies(t, X.parse_tree.proper_subformulas[1].val)
                            == H.top
                        ):
                            continue

                        proper_subformulas = [
                            phi,
                            AST_Node(
                                type="atom",
                                val=H.implies(
                                    t,
                                    X.parse_tree.proper_subformulas[1].val,
                                ),
                            ),
                        ]
                        new_signed_formula = Signed_Formula(
                            sign="F",
                            parse_tree=AST_Node(
                                type=X.parse_tree.type,
                                val=X.parse_tree.val,
                                proper_subformulas=proper_subformulas,
                            ),
                        )
                        new_world = gen.get_new_symbol()
                        new_relation = current_node.relation.union(
                            {f"{current_node.world}#{t.value}#{new_world}"}
                        )
                        n = Tableau_Node(
                            world=new_world,
                            relation=new_relation,
                            signed_formula=new_signed_formula,
                            parent=S[-1],
                        )
                        S[-1].children.append(n)
                        q.append(n)
                reactivate(current_node, q, H)

        update_closed(current_node, H)
    update_closed(tableau.root, H)

    if print:
        pt = PrettyPrintTree(
            lambda x: x.children,
            lambda x: str(x.signed_formula),
            lambda x: f"<{x.world}, {x.relation}>",
        )
        pt(tableau.root)
    return tableau


def isValid(phi: str, H: HeytingAlgebra):
    phi_parsed = parse_expression(phi)
    bounding_imp = AST_Node(
        type="binop",
        val="->",
        proper_subformulas=[AST_Node(type="atom", val=H.top), phi_parsed],
    )
    signed_bounding_imp = Signed_Formula("F", bounding_imp)

    tableau = construct_tableau(signed_bounding_imp, H)
    return tableau.isClosed()


def construct_counter_model(formula: str, H: HeytingAlgebra, tableau: Tableau = None):
    formula_parsed = parse_expression(formula)
    bounding_imp = AST_Node(
        type="binop",
        val="->",
        proper_subformulas=[AST_Node(type="atom", val=H.top), formula_parsed],
    )
    signed_bounding_imp = Signed_Formula("F", bounding_imp)
    if not tableau:
        tableau = construct_tableau(signed_bounding_imp, H)
    if tableau.isClosed():
        print("Tableau is closed")
        return
    open_branches = get_open_branches(tableau.root)
    if not open_branches:
        raise Exception("No open branches")
    S = open_branches[0]
    W = worlds(S)
    R_tmp = cons(S)
    prop_vars = re.findall(r"[p-z]\d*", formula)

    def get_bounds(S: list[Tableau_Node], prop_var: str, world: str):
        lbs = []
        ubs = []
        for s in S:
            if (
                s.signed_formula.parse_tree.proper_subformulas[1].val == prop_var
                and s.world == world
                and s.signed_formula.sign == "T"
                and isinstance(
                    s.signed_formula.parse_tree.proper_subformulas[0].val, TruthValue
                )
            ):
                lbs.append(s.signed_formula.parse_tree.proper_subformulas[0].val)
            if (
                s.signed_formula.parse_tree.proper_subformulas[0].val == prop_var
                and s.world == world
                and s.signed_formula.sign == "T"
                and isinstance(
                    s.signed_formula.parse_tree.proper_subformulas[1].val, TruthValue
                )
            ):
                ubs.append(s.signed_formula.parse_tree.proper_subformulas[1].val)

        return (lbs, ubs)

    valuation = {w: {p: None for p in prop_vars} for w in W}
    for w in W:
        for p in prop_vars:
            lbs, _ = get_bounds(S, p, w)
            sup = functools.reduce(
                H.join,
                lbs,
                H.bot,
            )
            valuation[w][p] = sup.value

    R = {u: {v: H.bot.value for v in W} for u in W}
    pattern = f"(.*)#(.*)#(.*)"
    for u, t, v in [
        (m.group(1), m.group(2), m.group(3))
        for r in R_tmp
        if (m := re.match(pattern, r))
    ]:
        R[u][v] = t

    return (W, R, valuation)


import networkx as nx
import matplotlib.pyplot as plt


def visualize_model(
    M: tuple[set[str], dict[str, dict[str, str]], dict[str, dict[str, str]]]
):
    G = nx.DiGraph()
    edges = M[1]

    # adding edges with weights
    for node, neighbours in edges.items():
        for neighbour, weight in neighbours.items():
            G.add_edge(node, neighbour, weight=weight)

    display_edges = [
        (u, v) for (u, v, d) in G.edges(data=True) if not d["weight"] == "0"
    ]

    pos = nx.spring_layout(G)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    for node, (x, y) in pos.items():
        plt.text(x, y, str(M[2][node]), fontsize=12, ha="right")

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=display_edges, width=3)

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.show()


if __name__ == "__main__":
    # expression = "(p -> (q -> p))"
    # expression = "[](p -> q) -> ([]p -> []q)"
    # expression = "a -> (((a -> <>p) & (1 -> []q)) -> <>(p & q))"
    expression = "<>p -> []p"
    # expression = "(p | (p -> 0))"
    # expression = "a -> (((a -> p) & (1 -> (p -> q))) -> q)"
    # expression = "(((a -> p) & (a -> (p -> q))) -> q)"
    signed_form = Signed_Formula("F", parse_expression(expression))

    bot = TruthValue("0")
    top = TruthValue("1")
    a = TruthValue("a")
    b = TruthValue("b")

    # Algebra 1
    # meetOp = {
    #     bot: {bot: bot, a: bot, b: bot, top: bot},
    #     a: {bot: bot, a: a, b: bot, top: a},
    #     b: {
    #         bot: bot,
    #         a: bot,
    #         b: b,
    #         top: b,
    #     },
    #     top: {
    #         bot: bot,
    #         a: a,
    #         b: b,
    #         top: top,
    #     },
    # }
    # ha = HeytingAlgebra({bot, a, b, top}, meetOp=meetOp)

    # Algebra 2
    order = {bot: {bot, a, top}, a: {a, top}, top: {top}}
    p = Poset({bot, a, top}, order=order)
    ha = HeytingAlgebra({bot, a, top}, poset=p)
    # tableau = construct_tableau(signed_form, ha)
    M = construct_counter_model(expression, ha)
    visualize_model(M)
    # print(f"{expression} is valid: {isValid(expression, ha)}")
    pass
