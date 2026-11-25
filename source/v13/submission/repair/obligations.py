"""Check narrow, inspectable semantic disagreements between verified branches."""
from dataclasses import dataclass
import re

from sqlglot import exp

from .structure import header_agreement, local, parts, parse, sources


ABSENCE = re.compile(r"\b(?:never|without|(?:do|does|did) not have|have no|has no|"
                     r"(?:do|does|did) not (?:have|contain|participate|play|take)|"
                     r"not (?:enrolled|assigned|used|involved|broadcast)|don.t have)\b")
COMPARISON = re.compile(r"\b(at least|at most|more than|greater than|fewer than|less than|exactly) "
                        r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\b")
EXTREME = re.compile(r"\b(older|younger|higher|lower|greater|less|more|after|before|larger|smaller)"
                     r"(?: than)?\s+(?:that of\s+)?(any|all|every|each)\b")
NUMBER_WORDS = dict(zip(("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
                        map(str, range(1, 11))))
OPERATORS = {
    "at least": exp.GTE, "at most": exp.LTE, "more than": exp.GT, "greater than": exp.GT,
    "fewer than": exp.LT, "less than": exp.LT, "exactly": exp.EQ,
}
COMPARATORS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


@dataclass(frozen=True)
class Request:
    text: str
    quoted: tuple
    boundary: tuple
    absence: bool
    quantifier: tuple | None
    direction: bool | None
    limit: int | None
    distinct: bool

    @classmethod
    def read(cls, question):
        quoted = re.findall(r"""(?<!\w)(?:'([^'\r\n]+)'|"([^"\r\n]+)")(?!\w)""", question)
        question = question.casefold()
        directions = re.findall(r"\b(ascending|descending|alphabetical)\b", question)
        direction = directions[0] == "descending" if len(set(directions)) == 1 else None
        top = re.search(r"\b(?:top|bottom|first|last)\s+(\d{1,2})\b", question)
        extreme = EXTREME.search(question)
        boundary = [(OPERATORS[operator], NUMBER_WORDS.get(number, number))
                    for operator, number in COMPARISON.findall(question)]
        return cls(question, tuple((a or b).strip().casefold() for a, b in quoted if a or b),
                   tuple(boundary), bool(ABSENCE.search(question)), extreme.groups() if extreme else None,
                   direction, int(top[1]) if top else None,
                   bool(re.search(r"\b(?:distinct|different|unique)\b", question)))


def predicate_nodes(tree):
    for clause in tree.find_all(exp.Where, exp.Having):
        yield from (node for node in clause.this.walk()
                    if node.find_ancestor(exp.Where, exp.Having) is clause)


def entity_absence(tree, predicates):
    if isinstance(tree, exp.Except):
        return True
    for node in predicates:
        if isinstance(node, exp.Not):
            child = node.this
            if isinstance(child, exp.Exists):
                sub = child.this
                inner = {t.alias_or_name.casefold() for t in sources(sub)} if isinstance(sub, exp.Select) else set()
                if not inner or any(field.table and field.table.casefold() not in inner for field in local(sub, exp.Column)):
                    return True
            if isinstance(child, exp.In) and child.args.get("query") is not None:
                return True
    for select in tree.find_all(exp.Select):
        left = {join.this.alias_or_name.casefold() for join in select.args.get("joins", [])
                if str(join.args.get("side", "")).upper() == "LEFT" and isinstance(join.this, exp.Table)}
        if left and any(isinstance(node, exp.Is) and isinstance(node.expression, exp.Null)
                        and isinstance(node.this, exp.Column) and node.this.table.casefold() in left
                        for node in predicates):
            return True
    return False


def join_signals(tree, schema):
    incorrect = impossible = 0
    for select in tree.find_all(exp.Select):
        aliases = {table.alias_or_name.casefold(): table.name.casefold() for table in sources(select)}
        for join in select.args.get("joins", []):
            on = join.args.get("on")
            if on is None:
                continue
            if isinstance(on, exp.EQ) and on.this == on.expression:
                impossible += 1
                continue
            for predicate in on.find_all(exp.EQ):
                left, right = predicate.this, predicate.expression
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    continue
                source, target = aliases.get(left.table.casefold()), aliases.get(right.table.casefold())
                if not source or not target or source == target:
                    continue
                edges = schema.edges.get(source, ()) if schema else ()
                candidates = [(a, b) for t, a, b in edges if t == target]
                if candidates and not any(a.casefold() == left.name.casefold() and b.casefold() == right.name.casefold()
                                          for a, b in candidates):
                    incorrect += 1
    return -min(2, incorrect), -min(2, impossible)


def signals(sql, request, example, schema):
    tree = parse(sql)
    if tree is None:
        return {}
    top = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if top is None:
        return {}
    predicates = list(predicate_nodes(tree))
    grounded, tautological = join_signals(tree, schema)
    result = {"tautological_join": tautological}
    if schema:
        result["relationship_key"] = grounded
    agreement = header_agreement(top, example["columns"])
    result["aligned_projection"] = int(agreement >= .99)
    if request.quoted:
        values = {node.this.strip("%").strip().casefold() for node in predicates
                  if isinstance(node, exp.Literal) and node.is_string}
        result["quoted_filter"] = sum(literal in values for literal in set(request.quoted))
    bound_values = {node.this.strip("%").strip().casefold() for node in predicates
                    if isinstance(node, exp.Literal) and node.is_string and len(node.this.strip("%")) >= 3
                    and re.search(r"[a-z]", node.this, re.I)}
    if bound_values:
        result["literal_grounding"] = sum(bool(re.search(r"\b" + re.escape(value) + r"\b", request.text))
                                          for value in bound_values)
    if request.absence:
        if entity_absence(tree, predicates):
            result["absence"] = 1
        elif any(isinstance(node, exp.NEQ) or (isinstance(node, exp.Not)
                 and isinstance(node.this, (exp.In, exp.Like))) for node in predicates):
            result["absence"] = -1
        else:
            result["absence"] = 0
    if request.boundary:
        satisfied = inconsistent = 0
        for operator, value in request.boundary:
            matching = [node for node in predicates if isinstance(node, COMPARATORS)
                        and isinstance(node.expression, exp.Literal) and not node.expression.is_string
                        and node.expression.this == value]
            if len(matching) == 1:
                satisfied += type(matching[0]) is operator
                inconsistent += type(matching[0]) is not operator
        result["numeric_boundary"] = satisfied - inconsistent
    order = tree.args.get("order")
    alphabet = re.search(r"\balphabetical(?:ly| order)?\s*(?:by|of)\s+([^.;?]+)", request.text)
    if alphabet:
        topic = parts(alphabet[1]) - {"a", "an", "the", "their", "its", "and", "each"}
        result["alphabetical_field"] = int(bool(order and any(
            parts(c.name) & topic and not parts(c.name) & {"id", "code", "key"}
            for c in order.find_all(exp.Column))))
    if request.direction is not None:
        result["explicit_order"] = (1 if bool(order.expressions[0].args.get("desc")) == request.direction else -1) if order else -1
    if request.limit is not None:
        limit = tree.args.get("limit")
        result["explicit_limit"] = int(bool(limit and isinstance(limit.expression, exp.Literal)
                                           and limit.expression.this == str(request.limit)))
    if request.quantifier:
        operator, quantifier = request.quantifier
        wants_max = (operator in ("older", "higher", "greater", "more", "after", "larger")) != (quantifier == "any")
        for comparison in predicates:
            if isinstance(comparison, (exp.GT, exp.LT, exp.GTE, exp.LTE)) and isinstance(comparison.expression, exp.Subquery):
                sub = comparison.expression.this
                if isinstance(sub, exp.Select) and len(sub.expressions) == 1:
                    value = sub.expressions[0].unalias()
                    result["quantified_extreme"] = (int(isinstance(value, exp.Max)) -
                                                    int(isinstance(value, exp.Min))) * (1 if wants_max else -1)
                    break
    if request.distinct:
        dist = top.args.get("distinct") or any(p.find(exp.Distinct) for p in top.expressions)
        result["explicit_distinct"] = int(bool(dist) or isinstance(tree, (exp.Intersect, exp.Except, exp.Union)))
    return result


def compare(challenger, incumbent):
    for name in sorted(challenger.keys() & incumbent.keys()):
        diff = challenger[name] - incumbent[name]
        if diff:
            yield name, 1 if diff > 0 else -1


def structural_margin(challenger, incumbent, weights, empty):
    """Count calibrated evidence once per obligation, regardless of AST size."""
    general = weights.get("all", {})
    contextual = weights.get("empty" if empty else "nonempty", {})
    return sum(direction * contextual.get(name, general.get(name, 0.0))
               for name, direction in compare(challenger, incumbent))


def relationship_refinement(incumbent, challenger, schema):
    """Approve an FK bridge only when it preserves the rest of the query."""
    old, new = parse(incumbent), parse(challenger)
    if not schema or not isinstance(old, exp.Select) or not isinstance(new, exp.Select):
        return 0
    if any(select.find(exp.Subquery, exp.SetOperation) for select in (old, new)):
        return 0
    tables = [{t.name.casefold() for t in sources(select)} for select in (old, new)]
    if not tables[0] or not tables[0] < tables[1] or len(tables[1] - tables[0]) > 2:
        return 0

    def signatures(query):
        aliases = {t.alias_or_name.casefold(): t.name.casefold() for t in sources(query)}

        def normalize(node):
            if isinstance(node, exp.Column) and node.table.casefold() in aliases:
                node.set("table", exp.to_identifier(aliases[node.table.casefold()]))
            return node

        def expr(value):
            return value.transform(normalize).sql(dialect="sqlite", normalize=True, comments=False)

        shape = [[expr(value.unalias()) for value in query.expressions]]
        for name in ("where", "group", "having", "order", "limit", "offset", "distinct"):
            shape.append(expr(query.args[name]) if query.args.get(name) else None)
        return shape

    if signatures(old) != signatures(new):
        return 0

    def linked(query):
        aliases = {t.alias_or_name.casefold(): t.name.casefold() for t in sources(query)}
        total = approved = 0
        for join in query.args.get("joins", []):
            comparison = join.args.get("on")
            if not isinstance(comparison, exp.EQ):
                return 0, 0
            left, right = comparison.this, comparison.expression
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                return 0, 0
            source, target = aliases.get(left.table.casefold()), aliases.get(right.table.casefold())
            if not source or not target or source == target:
                return 0, 0
            total += 1
            approved += any(t == target and a.casefold() == left.name.casefold() and
                            b.casefold() == right.name.casefold() for t, a, b in schema.edges.get(source, ()))
        return approved, total

    old_bound, old_joins = linked(old)
    new_bound, new_joins = linked(new)
    return int(0 < old_joins and old_bound < old_joins and new_bound == new_joins > old_joins)
