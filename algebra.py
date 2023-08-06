from collections import deque, defaultdict
from dataclasses import dataclass


class Poset:
    def __init__(self, elements: set, order=None):
        self.elements = elements
        if order == None:
            self.order = defaultdict(set)
        self.topsort = None

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

        if self.meetOp == None:
            self.meetOp = {a: {b: None for b in self.elements} for a in self.elements}
            if poset == None and joinOp == None:
                raise ValueError(
                    "At least one of meetOp, joinOp or poset must be passed in order to uniquely determine the bounded lattice"
                )
            self.deriveMeet()

        if self.joinOp == None:
            self.joinOp = {a: {b: None for b in self.elements} for a in self.elements}
            if poset == None and meetOp == None:
                raise ValueError(
                    "At least one of meetOp, joinOp or poset must be passed in order to uniquely determine the bounded lattice"
                )
            self.deriveJoin()

        if self.poset == None:
            self.derivePoset()

        if self.impliesOp == None:
            self.deriveImplies()

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
        return

    def meet(self, a, b):
        """Define your meet operation here"""
        pass

    def join(self, a, b):
        """Define your join operation here"""
        pass


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

    ha = HeytingAlgebra({bot, a, b, top}, meetOp=meetOp)
    print("done")
