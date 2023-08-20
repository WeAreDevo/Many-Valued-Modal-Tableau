from collections import deque, defaultdict
from dataclasses import dataclass
import functools


class Poset:
    def __init__(self, elements: set, order=None):
        self.elements = elements
        if order == None:
            self.order = defaultdict(set)
        self.topsort = None

    def leq(self, a, b):
        return b in self.order[a]

    def addEdge(self, u, v):
        self.order[u].add(v)

    def topologicalSortUtil(self, u, visited, stack):
        visited[u] = True
        for v in self.order[u]:
            if visited[v] == False:
                self.topologicalSortUtil(v, visited, stack)
        stack.appendleft(u)

    def getTopSort(self):
        if self.topsort == None:
            self.topologicalSort()
        return self.topsort

    def topologicalSort(self):
        # construct list topsort from elements with the property: if a \leq b then a occurs in topsort before b (converse does not neccesarily hold if order is not total)
        visited = {e: False for e in self.elements}
        stack = deque()
        for e in self.elements:
            if visited[e] == False:
                self.topologicalSortUtil(e, visited, stack)
        self.topsort = list(stack)

    def minimals(self, S: set):
        M = set()
        for x in self.topsort:
            if x in S and all(not self.leq(m, x) for m in M):
                M.add(x)
        return M

    def maximals(self, S: set):
        M = set()
        for x in reversed(self.topsort):
            if x in S and all(not self.leq(x, m) for m in M):
                M.add(x)
        return M


@dataclass(
    frozen=True
)  # Make immutable and so hashable. Thus, lookup into element sets and operation dicts are fast.
class TruthValue:
    value: str


class HeytingAlgebra:
    def __init__(
        self,
        elements: set[TruthValue],
        meetOp: dict[TruthValue, dict[TruthValue, TruthValue]] = None,
        joinOp: dict[TruthValue, dict[TruthValue, TruthValue]] = None,
        impliesOp: dict[TruthValue, dict[TruthValue, TruthValue]] = None,
        poset: Poset = None,
    ):
        self.elements = elements
        self.meetOp = meetOp
        self.joinOp = joinOp
        self.impliesOp = impliesOp
        self.poset = poset
        self.bot = None
        self.top = None

        if self.meetOp == None:
            if poset == None and joinOp == None:
                raise ValueError(
                    "At least one of meetOp, joinOp or poset must be passed in order to uniquely determine the bounded lattice"
                )
            self.meetOp = {a: {b: None for b in self.elements} for a in self.elements}
            self.deriveMeet()

        if self.joinOp == None:
            if poset == None and meetOp == None:
                raise ValueError(
                    "At least one of meetOp, joinOp or poset must be passed in order to uniquely determine the bounded lattice"
                )
            self.joinOp = {a: {b: None for b in self.elements} for a in self.elements}
            self.deriveJoin()

        if self.poset == None:
            self.derivePoset()

        self.__findBot()

        if self.impliesOp == None:
            self.impliesOp = {
                a: {b: None for b in self.elements} for a in self.elements
            }
            self.deriveImplies()

        self.__findTop()

    def __findBot(self):
        for e in self.elements:
            if self.poset.order[e] == self.elements:
                self.bot = e
                return

    def __findTop(self):
        self.top = self.implies(self.bot, self.bot)

    def derivePoset(self):
        poset = Poset(self.elements)
        if self.meetOp != None:
            for x in self.meetOp.keys():
                for y in self.meetOp[x].keys():
                    if self.meetOp[x][y] == x:
                        poset.addEdge(x, y)
            self.poset = poset

        elif self.joinOp != None:
            for x in self.meetOp.keys():
                for y in self.meetOp[x].keys():
                    if self.joinOp[x][y] == x:
                        poset.addEdge(y, x)
            self.poset = poset

    def deriveImplies(self):
        # a -> b = join{c | a meet c \leq b}
        for a in self.elements:
            for b in self.elements:
                self.impliesOp[a][b] = functools.reduce(
                    self.join,
                    [c for c in self.elements if self.poset.leq(self.meet(a, c), b)],
                    self.bot,
                )
        return

    def deriveJoin(self):
        if self.poset == None:
            self.derivePoset()

        # Now do something with the topological sort?
        t_sort = self.poset.getTopSort()
        order = self.poset.order
        for a in self.elements:
            for b in self.elements:
                if self.joinOp[a][b] != None:
                    continue
                for c in t_sort:
                    if c in order[a] and c in order[b]:
                        self.joinOp[a][b] = c
                        break

    def deriveMeet(self):
        if self.poset == None:
            self.derivePoset()

        # Now do something with the topological sort?
        t_sort = self.poset.getTopSort()
        t_sort = reversed(t_sort)
        order = self.poset.order
        for a in self.elements:
            for b in self.elements:
                if self.meetOp[a][b] != None:
                    continue
                for c in t_sort:
                    if a in order[c] and b in order[c]:
                        self.meetOp[a][b] = c
                        break

    def meet(self, a, b):
        return self.meetOp[a][b]

    def join(self, a, b):
        return self.joinOp[a][b]

    def implies(self, a, b):
        return self.impliesOp[a][b]


if __name__ == "__main__":
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

    joinOp = {
        top: {
            top: top,
            a: top,
            bot: top,
            b: top,
        },
        a: {
            top: top,
            a: a,
            bot: a,
            b: top,
        },
        bot: {
            top: top,
            a: a,
            bot: bot,
            b: b,
        },
        b: {
            top: top,
            a: top,
            bot: b,
            b: b,
        },
    }

    ha = HeytingAlgebra({bot, a, b, top}, joinOp=joinOp)
    print("done")
