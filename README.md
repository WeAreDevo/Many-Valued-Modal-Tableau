# Many-Valued-Modal-Tableau
This repo contains the implementation of a desicion procedure for checking the validity of many-valued modal logic formulas. It is based on expansions of  Fitting's work in [[1]](#1) and [[2]](#2)

## Getting Started
- Download this repo.
- [Install Miniconda](https://doi.org/10.1007/978-94-017-2794-5)
- Open a terminal or Anaconda Prompt window. Navigate to the root folder of this repo and execute the comand: ```conda env create -f environment.yml ```
- Activate the new environment by executing the commad: `conda activate mvml`
- Run the python file with `python main.py -e "<expression>"`, where `<expression>` is the propositional modal formula you wish to check is valid. 

### Note
`<expression>` should only contain well formed combinations of strings denoting:
  -  propositional variables, which must match the regex `[p-z]\d*` (i.e. begin with a letter between "p" and "z", followed by a string of zero or more decimal digits)
  - a truth value from the specified algebra (see configurations below)
  - a connective such as `"&"`, `"|"`, `"->"`, `"[]"`, `"<>"` (These corrrespond respectively to the syntactic objects $\land, \lor, \supset, \Box, \Diamond$ as presented in [[2]](#2))
  - Matching parentheses `"("`,`")"`.

If some matching parentheses are not present, the usual order of precedence for propositional formulas is assumed.
  

## Configurations
By default, the class of frames in which validity is checked is the class of all $\mathcal{H}$-frames[^1], where $\mathcal{H}$ is the three-valued heyting algebra. To specify another finite heyting algebra, create a json file in the `algebra_specs` directory with the following format:
```json
{
    "elements": ["<t1>","<t2>",...,"<tn>"],
    "order": {"<t1>": {"<t1_1>",...,"<t1_k1>"}, "<t2>": {"<t2_1>",...,"<t2_k2>"},...,"<tn>": {"<tn_1>",...,"<tn_kn">}},
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
- If $a \in H$, then $a=I($`"<ti>"`$)$ for some `"<ti>"` in `elements`.
- `"<ti>"` is in `order["<tk>"]` iff $I($`"<tk>"`$)  \leq I($`"<ti>"`$)$ 
- `meet["<ti>"]["<tk>"]=="<mi_k>"` iff $I($`"<mi_k>"`$) = I($`"<ti>"`$) \land I($`"<tk>"`$)$
- `join["<ti>"]["<tk>"]=="<ji_k>"` iff $I($`"<ji_k>"`$) = I($`"<ti>"`$) \lor I($`"<tk>"`$)$

For example, a json specification of the three-valued heyting algebra $(\{0,\frac{1}{2},1\}, \land, \lor, 0,1,\leq)$ with $I($`"0"`$)=0, I($`"a"`$)=\frac{1}{2}, I($`"1"`$)=1,$ would be as follows:

```json
{
    "elements": ["0","a","1"],
    "order": {"0": {"0","a","1"}, "a": {"a","1"}, "1": {"1"}},
    "meet": {
            "0": {"0": "0", "a": "0", "1": "0"},
            "a": {"0": "0", "a": "a", "1": "a"},
            "1": {"0": "0", "a": "a", "1": "1"}
        },
    "join": {
            "0": {"0": "0", "a": "a", "1": "1"},
            "a": {"0": "a", "a": "a", "1": "1"},
            "1": {"0": "1", "a": "1", "1": "1"}
        }
}
```

### Note
Only **one** of the `order`, `meet` or `join` fields needs to be specified and the other two can be left out of the `json`. If at least one of these is specified, the others can be uniquely determined. For example, the following is an acceptable specification of a four-valued Heyting algebra:

```json
{
    "elements": ["0","a","b","1"],
    "order": {"0": {"0","a","b","1"}, "a": {"a","1"}, "b": {"b","1"}, "1": {"1"}}
}
```


[^1]: See [[3]](#3) for appropriate definitions.

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