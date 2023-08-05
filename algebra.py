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
        for v in self.order[v]:
            if visited[v] == False:
                self.topologicalSortUtil(v, visited, stack)
        stack.appendleft(u)

    def getTopSort(self):
        if self.topsort == None:
            self.topologicalSort()
        return self.topsort

    def topologicalSort(self):
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

        if meetOp == None:
            if poset == None and joinOp == None:
                raise ValueError(
                    "At least one of meetOp, joinOp or poset must be passed in order to uniquely determine the bounded lattice"
                )
            self.deriveMeet()

        if joinOp == None:
            if poset == None and meetOp == None:
                raise ValueError(
                    "At least one of meetOp, joinOp or poset must be passed in order to uniquely determine the bounded lattice"
                )
            self.deriveJoin()

        if poset == None:
            self.derivePoset()

        if impliesOp == None:
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
    ha.derivePoset()
