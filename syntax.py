from ply import lex, yacc
from algebra import TruthValue


class AST_Node:
    def __init__(self, type, val=None, proper_subformulas=None):
        self.type = type
        if proper_subformulas:
            self.proper_subformulas = proper_subformulas
        else:
            self.proper_subformulas = []
        self.val = val
        self.expression = str(self)

    def __eq__(self, other):
        return (
            self.type == other.type
            and self.val == other.val
            and self.proper_subformulas == other.children
        )

    def __str__(self) -> str:
        if self.type == "atom":
            return str(self.val)
        if self.type == "unop":
            return f"{self.val}{str(self.proper_subformulas[0])}"
        if self.type == "binop":
            return f"({str(self.proper_subformulas[0])} {self.val} {str(self.proper_subformulas[1])})"


tokens = ("VAR", "BOX", "DIAMOND", "AND", "OR", "IMPLIES", "LPAREN", "RPAREN", "VALUE")

t_VAR = r"[p-z]\d*"
t_BOX = r"\[\]"
t_DIAMOND = r"<>"
t_AND = r"&"
t_OR = r"\|"
t_IMPLIES = r"->"
t_LPAREN = r"\("
t_RPAREN = r"\)"

t_ignore = " \t"


def t_VALUE(t):
    r"[a-o01]\d*"
    t.value = TruthValue(t.value)
    return t


def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


lexer = lex.lex()

precedence = (
    ("left", "IMPLIES"),
    ("left", "OR"),
    ("left", "AND"),
    ("right", "DIAMOND", "BOX"),
)


def p_expression(p):
    """
    expression : expression AND expression
                | expression OR expression
                | expression IMPLIES expression
                | BOX expression
                | DIAMOND expression
                | VAR
                | VALUE
    """
    if len(p) == 2:
        p[0] = AST_Node(type="atom", val=p[1], proper_subformulas=[])
    elif len(p) == 3:
        p[0] = AST_Node(type="unop", val=p[1], proper_subformulas=[p[2]])
    else:
        # TODO
        # if isinstance(p[1], Semantics) and isinstance(p[3], Semantics):
        #     #perform algebraic operation denoted by p[2] and set p[0] to the result
        p[0] = AST_Node(type="binop", val=p[2], proper_subformulas=[p[1], p[3]])


def p_expression_paren(p):
    """
    expression : LPAREN expression RPAREN
    """
    p[0] = p[2]


def p_error(p):
    print(f"Syntax error at '{p.value}'")


parser = yacc.yacc()


def parse_expression(expression):
    return parser.parse(expression, lexer=lexer)


if __name__ == "__main__":
    expression = "[](p & q) | q ->a"
    parsed_formula1 = parse_expression(expression)
    s = str(parsed_formula1)
    parsed_formula2 = parse_expression("((([]p) & p)  | q) ->a")

    from PrettyPrint import PrettyPrintTree

    pt = PrettyPrintTree(lambda x: x.proper_subformulas, lambda x: x.val)
    pt(parsed_formula1)
    pt(parsed_formula2)
    print(parsed_formula1 == parsed_formula2)
