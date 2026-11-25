"""Make inspectable, three-valued claims about a question and a checked query."""
from dataclasses import dataclass
import re

from sqlglot import exp

from .obligations import entity_absence, predicate_nodes
from .skills import leaves
from .structure import key, local, parse, parts, sources, sql_text


CONFLICT, UNKNOWN, SATISFIED = -1, 0, 1
NAME = {"name", "title", "label"}
DESCRIPTOR = {"type", "category", "description", "class", "status", "kind"}
STOP = {"a", "an", "all", "the", "their", "its", "each", "total", "distinct", "different", "unique"}


@dataclass(frozen=True)
class Claim:
    state: int
    reason: str


def scope(query, schema):
    return {t.alias_or_name.casefold(): t.name.casefold() for t in sources(query)
            if t.name.casefold() in schema.tables}


def owner(field, aliases, schema):
    if not isinstance(field, exp.Column):
        return None
    name = field.name.casefold()
    if field.table:
        relation = aliases.get(field.table.casefold())
        return relation if name in schema.columns.get(relation, {}) else None
    matches = [relation for relation in aliases.values() if name in schema.columns[relation]]
    return matches[0] if len(matches) == 1 else None


def relation_for(phrase, schema, allowed=None):
    terms = parts(phrase) - STOP
    if "people" in terms:
        terms.add("person")
    hits = [(len(words), table) for table, name in schema.tables.items()
            if (allowed is None or table in allowed) and (words := parts(name)) and words <= terms]
    if not hits:
        return None
    best = max(size for size, _ in hits)
    unique = [table for size, table in hits if size == best]
    return unique[0] if len(unique) == 1 else None


def explicit_entity_value(request, value, schema):
    for match in re.finditer(r"(?<!\w)" + re.escape(value) + r"(?!\w)", request.text):
        prefix = request.text[max(0, match.start() - 68):match.start()].rstrip(" '\"“”‘’")
        anchor = re.search(r"(?:\b(?:the|a|an|of|for|in|at)\s+|^)([\w -]{2,40})$", prefix)
        phrase = anchor[1] if anchor else prefix.rsplit(" ", 1)[-1]
        named = re.search(r"\b(?:named|called)\s*$", phrase)
        if named:
            phrase = phrase[:named.start()].strip()
        if parts(phrase) & DESCRIPTOR or parts(phrase) & {"code", "id", "number", "year"}:
            continue
        # Only use the closest words. Longer matches disambiguate base and subtype tables.
        words = phrase.split()
        for start in range(max(0, len(words) - 4), len(words)):
            relation = relation_for(" ".join(words[start:]), schema)
            if relation and (named or parts(schema.tables[relation]) & parts(" ".join(words[-2:]))):
                names = [meta["name"] for meta in schema.columns[relation].values()
                         if parts(meta["name"]) & NAME and not parts(meta["name"]) & DESCRIPTOR]
                if names:
                    return relation
    return None


def literal_claims(tree, request, schema):
    found = {}
    for node in predicate_nodes(tree):
        if not isinstance(node, (exp.EQ, exp.Like)):
            continue
        left, right = node.this, node.expression
        if isinstance(left, exp.Literal):
            left, right = right, left
        if not (isinstance(left, exp.Column) and isinstance(right, exp.Literal) and right.is_string):
            continue
        value = right.this.strip("%").strip().casefold()
        if len(value) < 3 or not re.search(r"[a-z]", value):
            continue
        relation = explicit_entity_value(request, value, schema)
        if relation is None:
            continue
        query = node.find_ancestor(exp.Select)
        bound = owner(left, scope(query, schema), schema) if query is not None else None
        words = parts(left.name)
        if bound == relation and words & NAME and not words & DESCRIPTOR:
            state = SATISFIED
        elif bound and (words & DESCRIPTOR or bound != relation and words & NAME):
            state = CONFLICT
        else:
            state = UNKNOWN
        found.setdefault(f"entity_name:{value}", []).append(Claim(state, f"named value is bound to {left.sql()}"))
    return {name: claims[0] for name, claims in found.items()
            if len({claim.state for claim in claims}) == 1}


def unique_key(table, name, schema):
    keys = [meta["name"].casefold() for meta in schema.columns.get(table, {}).values() if meta["pk"]]
    return keys == [name.casefold()]


def entity_key(field, query, target, schema):
    relation = owner(field, scope(query, schema), schema)
    if relation is None:
        return False
    if relation == target:
        return unique_key(target, field.name, schema)
    return any(other == target and on.casefold() == field.name.casefold() and unique_key(target, key, schema)
               for other, on, key in schema.edges[relation])


def fanout(query, target, schema):
    aliases = scope(query, schema)
    if len(aliases) <= 1:
        return False
    # The non-unique endpoint of a declared edge can repeat the target entity.
    for join in query.args.get("joins", []):
        condition = join.args.get("on")
        if not isinstance(condition, exp.EQ):
            continue
        for a, b in ((condition.this, condition.expression), (condition.expression, condition.this)):
            left, right = owner(a, aliases, schema), owner(b, aliases, schema)
            if left != target or right is None or left == right:
                continue
            if unique_key(target, a.name, schema) and not unique_key(right, b.name, schema) and any(
                    t == right and c.casefold() == a.name.casefold() and d.casefold() == b.name.casefold()
                    for t, c, d in schema.edges[target]):
                return True
    return False


def count_claim(query, request, schema):
    phrase = re.search(r"\b(?:how many|number of|count of)\s+([^,.?]{2,65})", request.text)
    if not phrase:
        return None
    subject = re.split(r"\b(?:who|that|which|have|has|with|are|were|did|from|in|for|by|per)\b", phrase[1], 1)[0]
    target = relation_for(subject, schema)
    counts = [p.unalias() for p in query.expressions if isinstance(p.unalias(), exp.Count)]
    if target is None or len(counts) != 1:
        return None
    count = counts[0]
    field = count.this
    distinct_key = isinstance(field, exp.Distinct) and len(field.expressions) == 1 and entity_key(
        field.expressions[0], query, target, schema)
    if distinct_key:
        return Claim(SATISFIED, f"distinct count uses the key of {schema.tables[target]}")
    if request.distinct and isinstance(field, exp.Distinct):
        return None
    if not isinstance(field, exp.Distinct) and fanout(query, target, schema):
        return Claim(CONFLICT, f"count sees repeated {schema.tables[target]} entities after a one-to-many join")
    if len(sources(query)) == 1 and sources(query)[0].name.casefold() == target and (
            isinstance(field, exp.Star) or entity_key(field, query, target, schema)):
        return Claim(SATISFIED, f"count is evaluated once per {schema.tables[target]} row")
    return None


def grouping_claim(query, request, schema):
    match = re.search(r"\b(?:for each|for every|per|grouped by)\s+([^,.?]{2,50})", request.text)
    if not match or not any(p.find(exp.AggFunc) for p in query.expressions):
        return None
    phrase = re.split(r"\b(?:show|find|list|give|with|of|that|who)\b", match[1], 1)[0].strip()
    relation = relation_for(phrase, schema)
    if not relation or not schema.primary(relation):
        return None
    group = query.args.get("group")
    if group is None:
        return Claim(CONFLICT, f"query aggregates without grouping by {schema.tables[relation]}")
    names = [p for p in group.expressions if isinstance(p, exp.Column)]
    if len(names) != len(group.expressions):
        return None
    if len(names) == 1 and entity_key(names[0], query, relation, schema):
        return Claim(SATISFIED, f"group key identifies each {schema.tables[relation]} entity")
    if names and all(owner(field, scope(query, schema), schema) not in (None, relation) for field in names):
        if request.text.find("grouped by") >= 0 or fanout(query, relation, schema):
            return Claim(CONFLICT, f"groups are keyed by a different relation from {schema.tables[relation]}")
    return None


def sorting_claim(query, request, schema):
    clause = re.search(r"\b(?:order(?:ed)?|sort(?:ed)?|alphabetically)\s+by\s+(?:the\s+)?([\w -]{2,55})", request.text)
    if not clause:
        return None
    phrase = re.split(r"\b(?:in|from|and|ascending|descending|order|alphabetically)\b", clause[1], 1)[0]
    terms = parts(phrase) - STOP
    fields = {(table, col.casefold()) for table, columns in schema.columns.items() for meta in columns.values()
              if (col := meta["name"]) and parts(col) == terms}
    if not fields:
        return None
    order = query.args.get("order")
    if order is None:
        return Claim(CONFLICT, "explicit sort field is absent")
    first = order.expressions[0].this
    if isinstance(first, exp.Column) and (owner(first, scope(query, schema), schema), first.name.casefold()) in fields:
        return Claim(SATISFIED, f"first sort key is the requested field {first.sql()}")
    if isinstance(first, exp.Column) and owner(first, scope(query, schema), schema):
        return Claim(CONFLICT, f"first sort key {first.sql()} differs from the explicitly named field")
    return None


def both_claim(tree, request):
    if not re.search(r"\bboth\b[^.!?]{1,85}\band\b", request.text):
        return None
    if isinstance(tree, exp.Intersect) and tree.this != tree.expression:
        return Claim(SATISFIED, "intersection requires both separate conditions")
    if not isinstance(tree, exp.Select) or tree.find(exp.Having, exp.Group):
        return None
    where = tree.args.get("where")
    rule = where.this if where else None
    if isinstance(rule, exp.And) and all(isinstance(side, (exp.In, exp.Exists)) for side in (rule.this, rule.expression)):
        if rule.this == rule.expression:
            return Claim(CONFLICT, "both membership conditions repeat the same subquery")
        if isinstance(rule.this, exp.In) and isinstance(rule.expression, exp.In) and rule.this.this == rule.expression.this:
            left, right = (side.args.get("query") for side in (rule.this, rule.expression))
            wanted = membership_words(request)
            if wanted and all(isinstance(value, exp.Subquery) and isinstance(value.this, exp.Select) for value in (left, right)):
                criteria = [side.this.args.get("where") for side in (left, right)]
                if all(condition and isinstance(condition.this, exp.EQ) for condition in criteria):
                    a, b = (condition.this for condition in criteria)
                    if a.this == b.this and all(isinstance(p.expression, exp.Literal) for p in (a, b)) and {
                            a.expression.this.casefold(), b.expression.this.casefold()} == set(wanted):
                        return Claim(SATISFIED, "separate membership queries each test one requested value")
        return None
    if isinstance(rule, exp.In) and isinstance(rule.this, exp.Column) and len(rule.expressions) == 2:
        values = rule.expressions
    elif isinstance(rule, (exp.Or, exp.And)) and all(isinstance(p, exp.EQ) for p in (rule.this, rule.expression)):
        if rule.this.this != rule.expression.this:
            return None
        values = [rule.this.expression, rule.expression.expression]
    else:
        return None
    if all(isinstance(value, exp.Literal) and re.search(
            r"(?<!\w)" + re.escape(value.this.casefold()) + r"(?!\w)", request.text) for value in values):
        if values[0] != values[1]:
            return Claim(CONFLICT, "one row-level value predicate cannot express two independent memberships")
    return None


def membership_words(request):
    values = re.search(r"\bboth\s+['\"]?([a-z][a-z-]{1,23})['\"]?\s+and\s+['\"]?([a-z][a-z-]{1,23})['\"]?(?=\s*(?:[?.!]|$))",
                       request.text)
    return values.groups() if values and values[1] != values[2] and not set(values.groups()) & STOP else None


def separate_memberships(tree, request, schema):
    """Use two independently filtered copies of an already bound membership query."""
    words = membership_words(request)
    condition = tree.args.get("where")
    rule = condition.this if condition else None
    if not words or not isinstance(rule, exp.And) or rule.this != rule.expression or not isinstance(rule.this, exp.In):
        return None
    inner = rule.this.args.get("query")
    if not isinstance(inner, exp.Subquery) or not isinstance(inner.this, exp.Select):
        return None
    where = inner.this.args.get("where")
    nested = where.this if where else None
    if not isinstance(nested, exp.In) or not isinstance(nested.this, exp.Column) or len(nested.expressions) != 2:
        return None
    if not all(isinstance(value, exp.Literal) and value.is_string for value in nested.expressions):
        return None
    if owner(nested.this, scope(inner.this, schema), schema) is None:
        return None
    selections = []
    for value in words:
        side = rule.this.copy()
        source = side.args["query"].this
        source.where(exp.EQ(this=exp.Lower(this=nested.this.copy()), expression=exp.Literal.string(value)),
                     append=False, copy=False)
        selections.append(tree.copy().where(side, append=False, copy=False))
    if any(tree.args.get(clause) for clause in ("order", "limit", "offset", "with", "with_")):
        return None
    return exp.Intersect(this=selections[0], expression=selections[1])


def restriction_repairs(query, request, schema):
    """Remove a redundant restriction from an explicitly distinct entity count."""
    if not isinstance(query, exp.Select) or not request.distinct or not re.search(r"\b(?:how many|number of|count of)\b", request.text):
        return
    if len(query.expressions) != 1 or any(query.args.get(tag) for tag in ("group", "having", "order", "limit")):
        return
    count = query.expressions[0].unalias()
    if not isinstance(count, exp.Count) or not isinstance(count.this, exp.Distinct):
        return
    if len(count.this.expressions) != 1 or not isinstance(count.this.expressions[0], exp.Column):
        return
    mentioned = parts(request.text)
    for index, join, alias, belongs in leaves(query, schema):
        relation = join.this.name.casefold()
        if relation not in schema.tables or parts(schema.tables[relation]) <= mentioned or join.args.get("side") or join.args.get("kind"):
            continue
        if any(belongs(field, alias) for field in local(query, exp.Column)
               if field.find_ancestor(exp.Join) is None):
            continue
        changed = query.copy()
        changed.set("joins", [j for i, j in enumerate(changed.args["joins"]) if i != index])
        yield changed


def equivalent_after_removing_restriction(old, new, request, schema):
    if schema is None:
        return False
    candidate = key(new)
    return any(key(sql_text(tree)) == candidate for tree in restriction_repairs(parse(old), request, schema))


def relationship_claim(query, schema):
    supported, conflicting = 0, 0
    aliases = scope(query, schema)
    for join in query.args.get("joins", []):
        check = join.args.get("on")
        if not isinstance(check, exp.EQ):
            continue
        left, right = check.this, check.expression
        a, b = owner(left, aliases, schema), owner(right, aliases, schema)
        if not a or not b or a == b:
            continue
        if any(t == b and x.casefold() == left.name.casefold() and y.casefold() == right.name.casefold()
               for t, x, y in schema.edges[a]):
            supported += 1
            continue
        route = schema.path([a], b)
        id_words = {"id", "key", "code", "number"}
        left_id, right_id = bool(parts(left.name) & id_words), bool(parts(right.name) & id_words)
        if route and (left_id != right_id or len(route) == 1 and unique_key(a, left.name, schema)
                      and unique_key(b, right.name, schema)):
            conflicting += 1
    if conflicting:
        return Claim(CONFLICT, "join compares the wrong key roles where a declared path exists")
    if supported and supported == len(query.args.get("joins", [])):
        return Claim(SATISFIED, "join equalities follow the declared keys")
    return None


def claims(sql, request, schema):
    tree = parse(sql)
    if tree is None or schema is None:
        return {}
    result = literal_claims(tree, request, schema)
    if isinstance(tree, exp.Select):
        for tag, check in (("count_grain", count_claim), ("group_key", grouping_claim), ("sort_field", sorting_claim)):
            claim = check(tree, request, schema)
            if claim is not None:
                result[tag] = claim
        relation = relationship_claim(tree, schema)
        if relation:
            result["join_key"] = relation
        if next(restriction_repairs(tree, request, schema), None) is not None:
            result["count_restriction"] = Claim(CONFLICT, "an unrequested joined relation restricts the distinct count")
    both = both_claim(tree, request)
    if both:
        result["independent_membership"] = both
    if request.absence:
        predicates = list(predicate_nodes(tree))
        if entity_absence(tree, predicates):
            result["entity_absence"] = Claim(SATISFIED, "absence is tested over an entity relation")
        elif isinstance(tree, exp.Select) and tree.args.get("joins") and any(
                isinstance(node, exp.NEQ) or isinstance(node, exp.Not) and isinstance(node.this, exp.Like)
                for node in predicates):
            result["entity_absence"] = Claim(CONFLICT, "negation filters joined rows rather than excluding related entities")
    return result


def improvements(previous, proposed):
    return {name for name in previous.keys() & proposed.keys()
            if previous[name].state == CONFLICT and proposed[name].state == SATISFIED}


def family(name):
    return name.split(":", 1)[0]
