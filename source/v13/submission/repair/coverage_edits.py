"""Ground filter and relation repairs in the current question and schema."""
import re

from sqlglot import exp

from .obligations import Request, predicate_nodes
from .semantic import owner, scope, separate_memberships
from .skills import Edit, conditions, leaves
from .structure import column, identifier, local, parts, sources


GENERIC = {"id", "code", "key", "number", "name", "value", "description", "detail", "type", "date"}


def vocabulary(text):
    words = parts(text)
    return words | {word[:-3] for word in words if word.endswith("ing") and len(word) > 6}


def role_score(field, relation, near, question):
    words = parts(field)
    table_words = parts(relation)
    specifics = words - GENERIC - table_words
    # A relation name alone is weak evidence for one of its fields.
    if not specifics and words & {"color", "status", "role", "category"}:
        specifics = words & {"color", "status", "role", "category"}
    score = 2.3 * len(specifics & near) + .48 * len(specifics & question)
    if words & near & GENERIC:
        score += .95
    if table_words and table_words <= near:
        score += 1.05
    if words & {"id", "code", "key"} and not words & near & {"id", "code", "key"}:
        score -= 2.2
    return score


def field_for_literal(query, field, value, schema, intent):
    literal = value.this.strip("%").strip().casefold()
    if not literal or len(literal) > 72:
        return None
    named = re.search(r"(?<!\w)" + re.escape(literal) + r"(?!\w)", intent.text)
    if named is None:
        return None
    near = vocabulary(intent.text[max(0, named.start() - 58):named.start()])
    question = vocabulary(intent.text)
    aliases = scope(query, schema)
    current = owner(field, aliases, schema)
    if current is None:
        return None
    old = role_score(field.name, schema.tables[current], near, question)
    candidates = []
    for relation, fields in schema.columns.items():
        route = schema.path(aliases.values(), relation)
        if route is None or len(route) > 2:
            continue
        for meta in fields.values():
            name = meta["name"]
            if (relation, name.casefold()) == (current, field.name.casefold()):
                continue
            terms = parts(name)
            if not terms & near or not (terms - GENERIC) & near and not parts(schema.tables[relation]) <= near:
                continue
            if terms & {"id", "key", "code"} and not terms & near & {"id", "key", "code"}:
                continue
            score = role_score(name, schema.tables[relation], near, question) - .55 * len(route)
            if score >= 2.25 and score >= old + 1.20:
                candidates.append((score, relation, name))
    candidates.sort(reverse=True)
    if not candidates or len(candidates) > 1 and candidates[0][0] - candidates[1][0] < .40:
        return None
    return candidates[0]


def grounded_predicate(tree, schema, intent):
    """Rebind one explicit equality to a nearby role, following up to two foreign keys."""
    matches = []
    for index, node in enumerate(predicate_nodes(tree)):
        if not isinstance(node, (exp.EQ, exp.NEQ, exp.Like)):
            continue
        value, field = node.expression, node.this
        if not isinstance(value, exp.Literal) or not isinstance(field, exp.Column):
            continue
        query = node.find_ancestor(exp.Select)
        if query is None or query.args.get("with_"):
            continue
        result = field_for_literal(query, field, value, schema, intent)
        if result:
            matches.append((result, index))
    for (_, relation, name), index in sorted(matches, reverse=True)[:2]:
        copy = tree.copy()
        target = list(predicate_nodes(copy))[index]
        select = target.find_ancestor(exp.Select)
        if not schema.attach(select, relation):
            continue
        alias = next(table.alias_or_name for table in sources(select) if table.name.casefold() == relation)
        target.set("this", column(name, alias))
        yield Edit("grounded_clause", copy, "bind a requested literal to its named field through the schema", .88)


def explicit_sort(query, schema, intent):
    text = re.search(r"\b(?:order(?:ed)?|sort(?:ed)?)\s+by\s+(?:the\s+)?([^.!?]{2,55})", intent.text)
    if not text or "number of" in text[1] or not isinstance(query, exp.Select):
        return
    current = query.args.get("order")
    if current and current.find(exp.AggFunc) or query.args.get("group"):
        return
    phrases = re.split(r"(?:,\s*|\s+and\s+|\s+then\s+)", text[1])[:3]
    fields = []
    for phrase in phrases:
        terms = parts(re.split(r"\bin\s+(?:ascending|descending)\b", phrase, 1)[0]) - {"ascending", "descending", "the", "order"}
        if not terms:
            return
        hits = [(len(parts(meta["name"]) - terms), alias, meta["name"]) for alias, table in scope(query, schema).items()
                for meta in schema.columns[table].values() if terms <= parts(meta["name"]) and (terms - GENERIC or len(terms) > 1)]
        hits.sort()
        if not hits or len(hits) > 1 and hits[0][0] == hits[1][0]:
            return
        _, alias, field = hits[0]
        fields.append(exp.Ordered(this=column(field, alias), desc="descending" in phrase))
    changed = query.copy().order_by(*fields, append=False, copy=False)
    yield Edit("grounded_clause", changed, "order by the explicitly requested schema fields", .87)


def relation_absence(query, schema, intent):
    """Rebuild the missing relation as an anti-semijoin over its declared key path."""
    terms = re.search(r"\b(?:without|never|(?:do|does|did) not|don.t|doesn.t|not (?:used|participat\w*|enrolled))\b([^.!?]{2,70})",
                      intent.text)
    if not terms or any(query.args.get(name) for name in ("group", "having", "with_")):
        return
    if not sources(query) or query.args.get("joins") or re.search(r"['\"]|\b(?:which|whose|where|with)\b", terms[1]):
        return
    first = sources(query)[0]
    entity = first.name.casefold()
    if entity not in schema.tables:
        return
    suffix = parts(terms[1])
    targets = []
    for table, display in schema.tables.items():
        words = parts(display)
        if table == entity or not words or not words <= suffix:
            continue
        path = schema.path([entity], table)
        if path and len(path) <= 2:
            targets.append((len(words), -len(path), table, path))
    targets.sort(key=lambda item: item[:3], reverse=True)
    if not targets or len(targets) > 1 and targets[0][:2] == targets[1][:2]:
        return
    path = targets[0][3]
    kept = []
    for rule in conditions(query):
        if isinstance(rule, exp.Not) and isinstance(rule.this, (exp.In, exp.Exists)) and rule.this.find(exp.Select):
            continue
        if rule.find(exp.Select):
            return
        kept.append(rule.copy())
    _, initial, left, right = path[0]
    inner = exp.select("1").from_(exp.Table(this=identifier(schema.tables[initial]), alias=exp.TableAlias(this=identifier("related_0"))))
    join_key = exp.EQ(this=column(left, first.alias_or_name), expression=column(right, "related_0"))
    if len(path) == 2:
        _, target, left, right = path[1]
        table = exp.Table(this=identifier(schema.tables[target]), alias=exp.TableAlias(this=identifier("related_1")))
        inner.join(table, on=exp.EQ(this=column(left, "related_0"), expression=column(right, "related_1")), copy=False)
    inner.where(join_key, copy=False)
    changed = query.copy().where(exp.and_(*kept, exp.Not(this=exp.Exists(this=inner))), append=False, copy=False)
    yield Edit("relation_scope", changed, "exclude entities with a row in the explicitly named related population", .91)


def frequency_scope(query, intent):
    if not re.search(r"\b(?:order|sort)\w*\b[^.?!]{0,38}\b(?:number of|count of)\b", intent.text):
        return
    if not 0 < len(query.expressions) <= 3 or query.args.get("having") or query.find(exp.Subquery):
        return
    outputs = [node.unalias().copy() for node in query.expressions]
    if any(not isinstance(node, exp.Column) for node in outputs):
        return
    changed = query.copy()
    changed.set("distinct", None)
    changed.group_by(*outputs, append=False, copy=False)
    changed.order_by(exp.Ordered(this=exp.Count(this=exp.Star()), desc=intent.descending is True), append=False, copy=False)
    yield Edit("relation_scope", changed, "group the requested output and order by its population count", .92)


def split_exclusion(query, intent):
    exclusion = re.search(r"\b(?:except|excluding|other than)\b([^.?]{3,68})", intent.text)
    where = query.args.get("where")
    if not exclusion or where is None or any(query.args.get(name) for name in ("group", "having", "order", "limit")):
        return
    if query.find(exp.Subquery, exp.AggFunc):
        return
    rule = where.this.unnest()
    # A top-level exclusion under the right-hand branch of OR can narrow only that branch.
    if not isinstance(rule, exp.Or) or not isinstance(rule.expression.unnest(), exp.And):
        return
    right_pair = rule.expression.unnest()
    excluded = right_pair.expression.unnest()
    if not isinstance(excluded, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)):
        return
    field, target = excluded.this, excluded.expression
    if not isinstance(field, exp.Column) or not isinstance(target, exp.Literal):
        return
    if not set(parts(field.name)) & parts(exclusion[1]) or not re.search(
            r"(?<!\w)" + re.escape(target.this.casefold()) + r"(?!\w)", exclusion[1]):
        return
    positive = query.copy().where(exp.or_(rule.this, right_pair.this), append=False, copy=False)
    negative = query.copy().where(excluded.copy(), append=False, copy=False)
    yield Edit("relation_scope", exp.Except(this=positive, expression=negative),
               "apply the explicit exclusion to both sides of the disjunction", .94)


def count_fact(query, schema, intent):
    subject = re.search(r"\b(?:number of|count of)\s+([^,.?]{2,45})", intent.text)
    if not subject or not query.find(exp.Count) or query.find(exp.Subquery) or not query.args.get("joins"):
        return
    phrase = re.split(r"\b(?:for|per|by|in|that|with|which|where|and)\b", subject[1], 1)[0]
    terms = parts(phrase)
    matching = [table.name.casefold() for table in sources(query) if parts(table.name) == terms]
    if len(matching) != 1:
        return
    fact = matching[0]
    for index, join, alias, belongs in leaves(query, schema):
        words = parts(join.this.name)
        if (join.this.name.casefold() == fact or not words or words <= intent.words or
                join.args.get("side") or join.args.get("kind")):
            continue
        if any(belongs(field, alias) for field in local(query, exp.Column) if field.find_ancestor(exp.Join) is None):
            continue
        changed = query.copy()
        changed.set("joins", [entry for i, entry in enumerate(changed.args["joins"]) if i != index])
        yield Edit("relation_scope", changed, "count the named fact without an unrelated joined detail", .88)
        break


def propose(tree, schema, intent):
    if tree is None or tree.args.get("with_"):
        return
    yield from grounded_predicate(tree, schema, intent)
    if isinstance(tree, exp.Select):
        yield from explicit_sort(tree, schema, intent)
        membership = separate_memberships(tree, Request.read(intent.text), schema)
        if membership is not None:
            yield Edit("relation_scope", membership, "test two independent named memberships", .97)
        yield from split_exclusion(tree, intent)
        yield from relation_absence(tree, schema, intent)
        yield from frequency_scope(tree, intent)
        yield from count_fact(tree, schema, intent)
