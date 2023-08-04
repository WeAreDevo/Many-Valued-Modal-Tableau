from ply import lex, yacc


class AST_Node:
    def __init__(self, type, val=None, children=None):
        self.type = type
        if children:
            self.children = children
        else:
            self.children = []
        self.val = val


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
    r"[a-o]\d*"
    # t.value = Semantics(t.value)
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
        # if isinstance(p[1], Semantics):
        #     p[0] = p[1]
        # else:
        p[0] = AST_Node(type="VAR", val=p[1], children=[])
    elif len(p) == 3:
        p[0] = AST_Node(type="unop", val=p[1], children=[p[2]])
    else:
        # TODO
        # if isinstance(p[1], Semantics) and isinstance(p[3], Semantics):
        #     #perform algebraic operation denoted by p[2] and set p[0] to the result
        p[0] = AST_Node(type="binop", val=p[2], children=[p[1], p[3]])


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


expression = "[]p & p | q"
parsed_formula = parse_expression(expression)

from PrettyPrint import PrettyPrintTree

pt = PrettyPrintTree(lambda x: x.children, lambda x: x.val)
pt(parsed_formula)
