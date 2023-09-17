import json
import algebra


def load_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data


def read_algebra(file_path: str):
    spec: dict = load_json(file_path)
    if not "elements" in spec:
        raise KeyError('"elements" key not found in algebra specification.')
    elements = {algebra.TruthValue(e) for e in spec["elements"]}

    poset = meet = join = None
    if not ("order" in spec or "meet" in spec or "join" in spec):
        raise KeyError(
            'At least one of the keys "order", "meet" or "join" must be provided in tge algebra specification.'
        )

    if "order" in spec:
        order = {
            algebra.TruthValue(k): set([algebra.TruthValue(e) for e in v])
            for k, v in spec["order"].items()
        }
        poset = algebra.Poset(elements=elements, order=order)
    if "meet" in spec:
        meet = {
            algebra.TruthValue(x): {
                algebra.TruthValue(y): algebra.TruthValue(m) for y, m in v.items()
            }
            for x, v in spec["meet"].items()
        }
    if "join" in spec:
        join = {
            algebra.TruthValue(x): {
                algebra.TruthValue(y): algebra.TruthValue(j) for y, j in v.items()
            }
            for x, v in spec["join"].items()
        }

    return algebra.HeytingAlgebra(
        elements=elements, meetOp=meet, joinOp=join, poset=poset
    )
