# Many-Valued-Modal-Tableau
This repo contains the implementation of a desicion procedure for checking the validity of many-valued modal logic formulas. It is based on a generalisation of Fitting's prefixed tableau system presented in [*Proof Methods for Modal and Intuitionistic Logics*](https://doi.org/10.1007/978-94-017-2794-5).

## Getting Started
- Download this repo.
- [Install Miniconda](https://doi.org/10.1007/978-94-017-2794-5)
- Open a terminal or Anaconda Prompt window. Navigate to the root folder of this repo and execute the comand: ```conda env create -f environment.yml ```
- Activate the new environment by executing the commad: `conda activate mvml`
- Run the python file with `python main.py -e '<expression>'`, where `<expression>` is the propositional modal formula you wish to check is valid

## Configurations
By default, the class of frames in which validity is checked is the class of all $\mathcal{H}$-frames[^1], where $\mathcal{H}$ is the three-valued heyting algebra. To specify another finite heyting algebra, create a json file in the `algebra_specs` directory with the following format:
```json
{
    "elements": [<t1>,<t2>,...,<tn>]
    "meetOp": {
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
}
```
`<ti>` for $i \in \{1,\ldots,n\}$ should be a string denoting a truth value of the logic. Such a string must match the regex `[a-o0-9]\d*`. That is, it should be a string of decimal digits, or a letter between a and o (inclusive) followed by a string of decimal digits.


[^1]: See Fitting's [Many-Valued Modal Logics II](https://doi.org/10.3233/FI-1992-171-205) for appropriate definitions.



## TODO
- Check if finite specified algebra is 