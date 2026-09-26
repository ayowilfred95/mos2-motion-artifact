"""Small helpers around the COMSOL Java API as exposed by MPh/JPype."""
import re
import time


class PropertyError(RuntimeError):
    pass


def props(node):
    return [str(p) for p in node.properties()]


def setp(node, name, value):
    """node.set(name, value) with a readable error that lists the valid property names."""
    names = props(node)
    if name not in names:
        raise PropertyError(f"{node.tag()} has no property '{name}'. Available: {sorted(names)}")
    node.set(name, value)


def setp_first(node, candidates, value, required=True):
    """Set the first property in `candidates` that the node has. Returns the name used."""
    names = props(node)
    for c in candidates:
        if c in names:
            node.set(c, value)
            return c
    if required:
        raise PropertyError(f"{node.tag()}: none of {candidates} exist. Available: {sorted(names)}")
    return None


def setidx_first(node, candidates, value, index, required=True):
    names = props(node)
    for c in candidates:
        if c in names:
            node.setIndex(c, value, index)
            return c
    if required:
        raise PropertyError(f"{node.tag()}: none of {candidates} exist. Available: {sorted(names)}")
    return None


def find_prop(node, pattern):
    """Property names matching a regex (used for version-dependent names)."""
    return [p for p in props(node) if re.search(pattern, p, re.I)]


def box(comp, tag, dim, xmin, xmax, ymin=None, ymax=None, label=None):
    """Geometric Box selection (entities fully inside the box). Coordinates in metres."""
    s = comp.selection().create(tag, "Box")
    s.set("entitydim", dim)
    s.set("xmin", repr(float(xmin)))
    s.set("xmax", repr(float(xmax)))
    if ymin is not None:
        s.set("ymin", repr(float(ymin)))
        s.set("ymax", repr(float(ymax)))
    s.set("condition", "inside")
    if label:
        s.label(label)
    return s


def union_sel(comp, tag, dim, inputs, label=None):
    s = comp.selection().create(tag, "Union")
    s.set("entitydim", dim)
    s.set("input", list(inputs))
    if label:
        s.label(label)
    return s


def dataset_of_study(model, study_tag):
    """Tag of the solution dataset produced by a study (after it has been run)."""
    j = model.java
    sol_tags = [str(t) for t in j.sol().tags() if str(j.sol(t).study()) == study_tag]
    for d in j.result().dataset().tags():
        ds = j.result().dataset(d)
        try:
            if str(ds.getString("solution")) in sol_tags:
                return str(d)
        except Exception:
            continue
    raise RuntimeError(f"no dataset found for study {study_tag}")


def eval_global(model, exprs, dataset_tag, units=None):
    """
    Evaluate global expressions on every inner solution of a dataset.
    Returns array of shape (n_expr, n_inner).
    """
    import numpy as np
    j = model.java
    tag = "gev_tmp"
    num = j.result().numerical()
    out = []
    for k, e in enumerate(exprs):
        if tag in [str(t) for t in num.tags()]:
            num.remove(tag)
        ev = num.create(tag, "EvalGlobal")
        ev.set("data", dataset_tag)
        ev.set("expr", [e])
        if units:
            ev.set("unit", [units[k]])
        ev.set("innerinput", "all")
        # one expression at a time, so the array orientation of getData() does not matter
        out.append(np.array(ev.getData(), dtype=float).ravel())
        num.remove(tag)
    return np.array(out)


def eval_nodes(model, expr, dataset_tag, unit=None):
    """Field values at mesh nodes (used for 1D profiles). Returns (coords, values)."""
    import numpy as np
    j = model.java
    tag = "interp_tmp"
    num = j.result().numerical()
    if tag in [str(t) for t in num.tags()]:
        num.remove(tag)
    ev = num.create(tag, "Eval")
    ev.set("data", dataset_tag)
    ev.set("expr", [expr])
    if unit:
        ev.set("unit", [unit])
    ev.set("innerinput", "all")
    vals = np.array(ev.getData(), dtype=float)
    coords = np.array(ev.getCoordinates(), dtype=float)
    num.remove(tag)
    return coords, vals


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key):
        timer = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()

            def __exit__(self_inner, *a):
                timer.t[key] = timer.t.get(key, 0.0) + time.perf_counter() - self_inner.t0
        return _Ctx()
