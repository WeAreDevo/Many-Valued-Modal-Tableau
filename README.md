# Many-Valued Modal Tableau (mvmt)
## Description
This repo contains the implementation of a desicion procedure for checking the validity of many-valued modal logic formulas. It is based on expansions of  Fitting's work in [[1]](#1) and [[2]](#2). The proof of the correctness and termination of the procedure formed part of my master's dissertation. More technical detail's can be found in this workshop [paper](https://ceur-ws.org/Vol-3717/paper1.pdf).

## Installation
```bash
pip install mvmt
```

## Getting Started
If you just want to use the package,
`pip install mvmt` and use the modules provided. See the colab notebook for example usages.
<a target="_blank" href="https://colab.research.google.com/github/WeAreDevo/Many-Valued-Modal-Tableau/blob/main/mvmt_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Demo In Colab"/>
</a>

If you would like to edit and run the source code directly, see the [For Developers](#for-developers) section.


## Function Documentation

This section describes the useful functions provided by the package.
### `utils.construct_heyting_algebra`

#### Description

A useful wrapper function for creating a `algebra.HeytingAlgebra` object from a user supplied `json` specification of a Heyting algebra. See below for the convention used to specify algebras in a `json` form.

#### Parameters

- `file_path` (`str`, optional): File path to `json` file containing a specification of a Heyting algebra.
- `python_dict` (`dict`, optional): Python `dict` giving the `json` specification of a Heyting algebra. Default is `None`.

At least one of the arguments must be given. The specification should have the following format:
```
{
    "elements": ["<t1>","<t2>",...,"<tn>"],
    "order": {"<t1>": ["<t1_1>",...,"<t1_k1>"], "<t2>": ["<t2_1>",...,"<t2_k2>"],...,"<tn>": ["<tn_1>",...,"<tn_kn">]},
    "meet": {
            "<t1>": {"<t1>": "<m1_1>", "<t2>": "<m1_2>", ..., "<tn>": "<m1_n>"},
            "<t2>": {"<t1>": "<m2_1>", "<t2>": "<m2_2>", ..., "<tn>": "<m2_n>"},
            .
            .
            .
            "<tn>": {"<t1>": "<mn_1>", "<t2>": "<mn_2>", ..., "<tn>": "<mn_n>"},
        },
    "join": {
            "<t1>": {"<t1>": "<j1_1>", "<t2>": "<j1_2>", ..., "<tn>": "<j1_n>"},
            "<t2>": {"<t1>": "<j2_1>", "<t2>": "<j2_2>", ..., "<tn>": "<j2_n>"},
            .
            .
            .
            "<tn>": {"<t1>": "<jn_1>", "<t2>": "<jn_2>", ..., "<tn>": "<jn_n>"},
        }
}
```

Where each angle bracket string in the above should be replaced with a string denoting a truth value. Such a string must match the regex `[a-o0-9]\d*`. That is, it should be a string of decimal digits, or a letter between a and o (inclusive) followed by a string of decimal digits.

If we assume the json is intended to represent a heyting algebra $\mathcal{H}=(H,\land,\lor,0,1, \leq)$, and $I$ is the mapping from the strings denoting truth values to the actual truth values in $H$, then the json should be interpreted as follows:
- $a \in H$ iff $a=I$("&lt;ti&gt;") for some `"<ti>"` in `elements`.
- `"<ti>"` is in `order["<tk>"]` iff $I$("&lt;tk&gt;")  $\leq I$("&lt;ti&gt;")
- `meet["<ti>"]["<tk>"]=="<mi_k>"` iff $I$("&lt;mi_k&gt;") $= I$("&lt;ti&gt;") $\land I$("&lt;tk&gt;")
- `join["<ti>"]["<tk>"]=="<ji_k>"` iff $I$("&lt;ji_k&gt;") $= I$("&lt;ti&gt;") $\lor I$("&lt;tk&gt;")

For example, a specification of the three-valued Heyting algebra $(\{0,\frac{1}{2},1\}, \land, \lor, 0,1,\leq)$ with $I(\texttt{"0"})=0, I(\texttt{"a"})=\frac{1}{2}, I(\texttt{"1"})=1$ is given by:
```json
{
    "elements": ["0","a","1"],
    "order": {
        "0": ["0","a","1"],
        "a": ["a","1"],
        "1": ["1"]
    },
    "meet": {
        "0": {"0": "0","a": "0","1": "0"},
        "a": {"0": "0","a": "a","1": "a"},
        "1": {"0": "0","a": "a","1": "1"}
    },
    "join": {
        "0": {"0": "0","a": "a","1": "1"},
        "a": {"0": "a","a": "a","1": "1"},
        "1": {"0": "1","a": "1","1": "1"}
    }
}
```


#### Note:
Only **one** of the `order`, `meet` or `join` fields needs to be specified and the other two can be left out of the `json`. If at least one of these is specified, the others can be uniquely determined. See the example code.

#### Returns

(`algebra.HeytingAlgebra`) A `algebra.HeytingAlgebra` object storing the algebra represented by the inputed specification.

#### Example
```python
from mvmt import algebra, utils

# Specification of a four-valued Heyting algebra
spec = {
    "elements": ["0","a","b","1"],
    "order": {
      "0": ["0","a","b","1"], 
      "a": ["a","1"], 
      "b": ["b","1"], 
      "1": ["1"]}
}

# Example usage
H: algebra.HeytingAlgebra = utils.construct_heyting_algebra(python_dict=spec)
print(H.meet(algebra.TruthValue("a"),algebra.TruthValue("b")))
```
```plaintext
# Output
0
```

### `syntax.parse_expression`

#### Description

Produces the syntax parse tree of a modal formula.

#### Parameters

- `expression` (`str`, required): A modal formula.

`expression` must belong to the language generated by the grammar
```
expression ::= expression & expression
           | expression | expression
           | expression -> expression
           | [] expression
           | <> expression
           | (expression)
           | VAR
           | VALUE
```
`VAR` is a terminal symbol matching the regex `[p-z]\d*`, denoting a propositional variable.

`VALUE` is a terminal symbol matching the regex `[a-o0-9]\d*`. It denotes a truth value from the algebra, and as such it must be the same as one of the strings given in `elements` of the algebra specification.

#### Returns

(`syntax.AST_Node`) The root node of the parse tree.

#### Example
```python
from mvmt import syntax

# Example usage
parse_tree = syntax.parse_expression("((<>(p -> 0) -> 0) -> []p)")
print(parse_tree.proper_subformulas[1].val)
```
```plaintext
# Output
[]
```


### `tableau.isValid`

#### Description

Given a modal formuala $\varphi$ and Heyting algebra $\mathcal{H}$, checks if $\varphi$ is valid in the class of all $\mathcal{H}$-frames[^1].

Under the hood, it calls `tableau.construct_tableau` to construct a prefixed tableau for $\varphi$ and then returns `True` iff the constructed tableau is closed. `tableau.construct_tableau` is essentially the crux of this package.

#### Parameters

- `phi` (`str`, required): The modal formula $\varphi$, with syntax as described in the [`parameters`](#parameters-1) section of `syntax.parse_expression`.
- `H` (`algebra.HeytingAlgebra`, required) The Heyting algebra $\mathcal{H}$.

#### Returns

- (`boolean`) `True` if $\varphi$ is valid in the class of all $\mathcal{H}$-frames and `False` otherwise.
- (`tableau.Tableau`) The tableau constructed and used to determine validity.

#### Example
```python
from tableau import isValid

# Example usage
valid, tab = tableau.isValid("[]p -> p", H)
print(valid)
```
```plaintext
# Output
False
```


### `tableau.construct_counter_model`

#### Description

Given a modal formuala $\varphi$ and Heyting algebra $\mathcal{H}$, if possible produce an $\mathcal{H}-model$ in which $\varphi$ is is not globally true.
The model is read off of an open tableau for $\varphi$.

#### Parameters

- `phi` (`str`, required): The modal formula $\varphi$, with syntax as described in the [`parameters`](#parameters-1) section of `syntax.parse_expression`.
- `H` (`algebra.HeytingAlgebra`, required) The Heyting algebra $\mathcal{H}$.
- `tableau` (`tableau.Tableau`, optional) A tableau for $\varphi$. This is just used to prevent duplicate computations if the required tableau was already produced by `isValid`.

#### Returns

- (`algebra.HeytingAlgebra`) $\mathcal{H}$
- (`tuple[set[str], dict[TruthValue, dict[TruthValue, TruthValue]], dict[str, dict[str, TruthValue]]]`) The constructed counter $\mathcal{H}$-model.

#### Example
```python
from tableau import construct_counter_model

# Example usage
M = tableau.construct_counter_model("[]p -> p", H, tab)
print(M[1][1])
```
```plaintext
# Output
{'A': {'A': '0'}}
```

### `tableau.visualize_model`

#### Description

Produces a `matplotlib` visualizing a model returned by `tableau.construct_counter_model` as a labeled weighted directed graph.

#### Parameters

- `M` (`tuple[set[str], dict[TruthValue, dict[TruthValue, TruthValue]], dict[str, dict[str, TruthValue]]]`): An $\mathcal{H}$-model.

#### Returns
None

[^1]: See [[3]](#3) for appropriate definitions.
## For Developers
To run the code from source, follow these steps.
- Clone this repo.
```bash
git clone https://github.com/WeAreDevo/Many-Valued-Modal-Tableau.git
cd Many-Valued-Modal-Tableau
```
- Create a virtual enironment containing the required dependencies and activate it using either `conda` 
```bash
conda env create -f environment.yml
conda activate mvmt
```
or using `virtualenv`
```bash
pip install virtualenv
virtualenv mvmt
source mvmt/bin/activate
pip install -r requirements.txt
```

- You can now edit the source code and run it locally. For example, run `main.py`:
```bash
python main.py "<expression>" --algebra filename.json --print_tableau --display_model
``` 
`<expression>` is the propositional modal formula you wish to check is valid.
`filename.json` should be the name of a file stored in `algebra_specs` that contains a json specification of a Heyting algebra.
The `--print_tableau` flag enables printing the constructed tableau to the terminal, and the `--display_model` flag enables displaying a counter model (if it exists) in a seperate window.

E.g.
```bash
python main.py "[]p->p" --algebra three_valued.json --print_tableau --display_model
```


### Note
`<expression>` should only contain well formed combinations of strings denoting:
  -  propositional variables, which must match the regex `[p-z]\d*` (i.e. begin with a letter between "p" and "z", followed by a string of zero or more decimal digits)
  - a truth value from the specified algebra (see configurations below)
  - a connective such as `"&"`, `"|"`, `"->"`, `"[]"`, `"<>"` (These corrrespond respectively to the syntactic objects $\land, \lor, \supset, \Box, \Diamond$ as presented in [[2]](#2))
  - Matching parentheses `"("`,`")"`.

If some matching parentheses are not present, the usual order of precedence for propositional formulas is assumed.

## References
<a id="1">[1]</a> 
Fitting, M. (1983). Prefixed Tableau Systems. In: Proof Methods for Modal and Intuitionistic Logics. Synthese Library, vol 169. Springer, Dordrecht. https://doi.org/10.1007/978-94-017-2794-5_9

<a id="2">[2]</a> 
Fitting, M. (1995). Tableaus for Many-Valued Modal Logic. Studia Logica: An International Journal for Symbolic Logic, 55(1), 63–87. http://www.jstor.org/stable/20015807

<a id="3">[3]</a> 
Fitting, M. (1992). Many-valued modal logics II. Fundamenta Informaticae, 17, 55-73. https://doi.org/10.3233/FI-1992-171-205


## TODO
- Check if specified finite algebra is in fact a bounded distributive lattice (and hence Heyting algebra)
- Allow choice of stricter clsses of frames
