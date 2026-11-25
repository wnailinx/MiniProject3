"""Repair one diagnosed defect on a previously checked query."""
import re

from sqlglot import exp

from .obligations import entity_absence, predicate_nodes
from .skills import Edit, binding, conditions, leaves, select_headers
from .structure import column, identifier, local, parts, sources


def local_binding(select, schema, intent, feedback):
    error = (feedback.get("error") or {}).get("message", "")
    missing = re.search(r"(no such|ambiguous) column(?: name)?:\s*([^\s]+)", error, re.I)
    if missing is None:
        return
    _, _, name = missing[2].strip('"').rpartition(".")
    scopes = list(select.find_all(exp.Select))
    for index, scope in enumerate(scopes[1:]):
        if not any(c.name.casefold() == name.casefold() for c in local(scope, exp.Column)):
            continue
        for edit in binding(scope, schema, intent, feedback):
            changed = select.copy()
            list(changed.find_all(exp.Select))[index + 1].replace(edit.tree)
            yield Edit("inner_binding", changed, "repair the reported name in the inner SELECT", edit.confidence)
            return


def grounded_filter(select, sibling, schema, request, intent):
    """Bring one confirmed condition from a different branch into current aliases."""
    if not isinstance(sibling, exp.Select) or not request.quoted:
        return
    siblings = {t.alias_or_name.casefold(): t.name.casefold() for t in sources(sibling)}
    present = {t.name.casefold(): t.alias_or_name for t in sources(select)}
    if not set(present) & set(siblings.values()):
        return
    existing = conditions(select)
    literals = {n.this.strip("%").casefold() for old in existing for n in old.find_all(exp.Literal) if n.is_string}
    for predicate in conditions(sibling):
        if not isinstance(predicate, (exp.EQ, exp.Like, exp.NEQ, exp.In)) or predicate.find(exp.Select):
            continue
        values = {n.this.strip("%").casefold() for n in predicate.find_all(exp.Literal) if n.is_string}
        if not values & (set(request.quoted) - literals):
            continue
        changed = select.copy()
        condition = predicate.copy()
        for node in condition.find_all(exp.Column):
            owner = siblings.get(node.table.casefold())
            if owner is None and not node.table:
                names = [name for name in siblings.values() if node.name.casefold() in schema.columns.get(name, {})]
                if len(names) == 1:
                    owner = names[0]
            if owner is None or not schema.attach(changed, owner):
                break
            alias = next(t.alias_or_name for t in sources(changed) if t.name.casefold() == owner)
            node.set("table", identifier(alias))
        else:
            changed.where(condition, copy=False)
            yield Edit("restore_filter", changed, "recover a question-grounded filter from a checked sibling", .87)
            return


def aggregate_filter(select, schema, intent, example):
    """Compare the existing record value to an overall or per-group statistic."""
    if not re.search(r"\b(above|below|greater than|less than|higher than|lower than|older than|"
                     r"younger than|more than)\s+(?:the\s+)?average\b", intent.text):
        return
    tables = sources(select)
    if len(tables) != 1 or select.find(exp.Subquery):
        return
    chosen = tables[0]
    columns = schema.columns.get(chosen.name.casefold(), {})
    aggregates = list(select.find_all(exp.Avg))
    aligned = [node.this for node in aggregates if isinstance(node.this, exp.Column)
               and node.this.name.casefold() in columns]
    if re.search(r"\b(older|younger)\s+than\b", intent.text):
        aligned = [column(c["name"]) for c in columns.values() if parts(c["name"]) == {"age"}] or aligned
    if not aligned:
        aligned = [column(c["name"]) for c in columns.values() if parts(c["name"]) & intent.words
                   and re.search(r"int|real|float|decimal|numeric|double", c["type"], re.I) and not c["pk"]]
    if len({c.name.casefold() for c in aligned}) != 1:
        return
    name = aligned[0].name
    tree = select_headers(select, example["columns"], schema, intent)
    if tree is None or len(sources(tree)) != 1:
        return
    low = re.search(r"\b(below|less than|lower than|younger than)\s+(?:the\s+)?average\b", intent.text)
    operator = exp.LT if low else exp.GT
    inner_alias = "repair_average"
    if chosen.alias_or_name.casefold() == inner_alias:
        return
    source = chosen.copy()
    source.set("alias", exp.TableAlias(this=identifier(inner_alias)))
    average = exp.select(exp.Avg(this=column(name, inner_alias))).from_(source)
    comparison = operator(this=column(name, chosen.alias_or_name), expression=exp.Subquery(this=average))
    tree.where(comparison, copy=False)
    if tree.args.get("group") and re.search(r"\b(each|per)\b", intent.text):
        grouped = [node for node in tree.args["group"].expressions if isinstance(node, exp.Column)]
        if len(grouped) == 1 and grouped[0].name.casefold() in columns:
            correlated = tree.copy()
            scalar = correlated.find(exp.Subquery).this
            group_name = grouped[0].name
            scalar.where(exp.EQ(this=column(group_name, inner_alias),
                                expression=column(group_name, chosen.alias_or_name)), copy=False)
            yield Edit("correlated_average", correlated, "compare to the average of the same requested group", .92)
    yield Edit("correlated_average", tree, "compare the record to the overall average", .82)


def explicit_absence(select, schema, request):
    if not request.absence or entity_absence(select, list(predicate_nodes(select))):
        return
    tables = sources(select)
    joins = select.args.get("joins", [])
    if not joins or len(tables) != len(joins) + 1 or select.args.get("group"):
        return
    if any(join.args.get("on") is None for join in joins):
        return
    main = tables[0]
    main_schema = schema.columns.get(main.name.casefold(), {})
    qualifiers = {t.alias_or_name.casefold() for t in tables}
    output = [c for p in select.expressions for c in p.find_all(exp.Column)]
    if not output and not any(isinstance(p.unalias(), exp.Count) for p in select.expressions):
        return
    if any((c.table and c.table.casefold() != main.alias_or_name.casefold()) or
           c.name.casefold() not in main_schema for c in output):
        return
    keep, inside = [], []
    for value in conditions(select):
        columns = list(value.find_all(exp.Column))
        if not columns or any(not c.table or c.table.casefold() not in qualifiers for c in columns):
            return
        if all(c.table.casefold() == main.alias_or_name.casefold() for c in columns):
            keep.append(value.copy())
        elif isinstance(value, exp.NEQ):
            inside.append(exp.EQ(this=value.this.copy(), expression=value.expression.copy()))
        elif isinstance(value, exp.Not):
            inside.append(value.this.copy())
        elif isinstance(value, (exp.EQ, exp.In, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)):
            inside.append(value.copy())
        else:
            return
    child = exp.select("1").from_(joins[0].this.copy())
    child.set("joins", [exp.Join(this=j.this.copy(), on=j.args["on"].copy()) for j in joins[1:]])
    child.where(exp.and_(joins[0].args["on"].copy(), *inside), copy=False)
    changed = select.copy()
    changed.set("joins", None)
    changed.where(exp.and_(*keep, exp.Not(this=exp.Exists(this=child))), append=False, copy=False)
    yield Edit("absence_chain", changed, "express absence across the already bound relationship", .91)


def explicit_count_target(select, schema, intent):
    if not intent.count or not select.args.get("joins") or len(select.expressions) != 1 or select.args.get("group"):
        return None
    count = select.expressions[0].unalias()
    if not isinstance(count, exp.Count) or isinstance(count.this, exp.Distinct):
        return None
    phrase = re.search(r"\b(?:how many|number of|count of)\s+([^,.?]{2,70})", intent.text)
    if phrase is None:
        return None
    subject = re.split(r"\b(?:who|that|which|have|has|with|are|were|did|from|in|for)\b", phrase[1], maxsplit=1)[0]
    terms = parts(subject) - {"all", "the", "total", "unique", "distinct", "different"}
    terms = terms | ({"person"} if "people" in terms else set())
    candidates = [table for table in sources(select)
                  if parts(table.name) <= terms and schema.primary(table.name.casefold())]
    if not candidates:
        return None
    table = max(candidates, key=lambda t: len(parts(t.name)))
    primary = schema.primary(table.name.casefold())
    return table, primary


def cardinality_query(select, schema, intent):
    target = explicit_count_target(select, schema, intent)
    if target is None:
        return None
    table, primary = target
    query = select.copy()
    for name in ("order", "offset", "limit", "distinct"):
        query.set(name, None)
    query.set("expressions", [
        exp.Count(this=exp.Star()),
        exp.Count(this=exp.Distinct(expressions=[column(primary, table.alias_or_name)])),
    ])
    changed = select.copy()
    count = changed.expressions[0].unalias()
    count.set("this", exp.Distinct(expressions=[column(primary, table.alias_or_name)]))
    return query, changed


def remove_unused_leaf(select, schema):
    for index, join, alias, belongs in leaves(select, schema):
        outside = [value for name, value in select.args.items()
                   if name not in {"joins", "from", "from_"} and value is not None]
        expressions = [expr for item in outside for expr in (item if isinstance(item, list) else [item])
                       if isinstance(expr, exp.Expression)]
        if any(belongs(col, alias) for expr in expressions for col in expr.find_all(exp.Column)):
            continue
        select.set("joins", [entry for i, entry in enumerate(select.args["joins"]) if i != index])
        return True
    return False


def entity_name_filter(select, schema, intent, diagnosis):
    """Move a named value to the relation the question names."""
    if not (diagnosis.empty or diagnosis.rows == "fewer") or select.args.get("group"):
        return
    if len(sources(select)) < 2:
        return
    output_owners = {c.table.casefold() for p in select.expressions for c in p.find_all(exp.Column) if c.table}
    for index, predicate in enumerate(conditions(select)):
        if not isinstance(predicate, (exp.EQ, exp.Like)) or not isinstance(predicate.this, exp.Column):
            continue
        value = predicate.expression
        if not isinstance(value, exp.Literal) or not value.is_string:
            continue
        text = value.this.strip("%").casefold()
        if len(text) < 3 or not re.search(r"[a-z]", text):
            continue
        match = re.search(r"\b" + re.escape(text) + r"\b", intent.text)
        if match is None:
            continue
        anchor = parts(intent.text[max(0, match.start() - 25):match.start()])
        source_terms = parts(predicate.this.name)
        if not source_terms & {"name", "description", "type", "title", "id"}:
            continue
        possible = []
        for table in sources(select):
            table_words = parts(table.name)
            if not table_words or not table_words <= anchor or table.alias_or_name.casefold() == predicate.this.table.casefold():
                continue
            if output_owners and table.alias_or_name.casefold() not in output_owners:
                continue
            for meta in schema.columns.get(table.name.casefold(), {}).values():
                field = meta["name"]
                terms = parts(field)
                if not terms & {"name", "title"} or terms & {"type", "id", "description"}:
                    continue
                possible.append((len(table_words), -len(terms), table.alias_or_name, field))
        if not possible:
            continue
        _, _, alias, field = max(possible)
        changed = select.copy()
        node = conditions(changed)[index]
        node.set("this", column(field, alias))
        remove_unused_leaf(changed, schema)
        yield Edit("entity_literal", changed, "bind the named value on the requested entity", .90)
        return


def unneeded_filter(select, schema, intent, diagnosis):
    """An unrestricted enumeration does not need an unexplained scalar filter."""
    if diagnosis.empty or diagnosis.rows != "fewer" or not re.search(r"\b(?:all|every)\b", intent.text):
        return
    if re.search(r"\b(that|which|whose|who|where|with|without|between|greater|less|at least|only)\b", intent.text):
        return
    relations = sources(select)
    if len(relations) != 1 or select.args.get("group") or select.args.get("having"):
        return
    clauses = conditions(select)
    if len(clauses) != 1 or not isinstance(clauses[0], (exp.EQ, exp.Like)):
        return
    condition = clauses[0]
    if not isinstance(condition.expression, exp.Literal) or not condition.expression.is_string:
        return
    relation_words = parts(relations[0].name)
    if not relation_words or len(relation_words & intent.words) < min(2, len(relation_words)):
        return
    literal = condition.expression.this.strip("%").casefold()
    if re.search(r"""['"]""" + re.escape(literal) + r"""['"]""", intent.text):
        return
    changed = select.copy()
    changed.set("where", None)
    yield Edit("unrestricted_list", changed, "restore the explicitly requested unrestricted list", .82)


def alphabetical_field(select, schema, intent, ordered):
    if not ordered:
        return
    match = re.search(r"\balphabetical(?:ly| order)?\s*(?:by|of)\s+([^.;?]+)", intent.text)
    if match is None:
        return
    topic = parts(match[1]) - {"a", "an", "the", "their", "its", "and", "each"}
    if not topic:
        return
    old = select.args.get("order")
    if old and any(isinstance(c, exp.Column) and parts(c.name) & topic and not parts(c.name) & {"id", "code", "key"}
                   for c in old.find_all(exp.Column)):
        return
    tables = {t.name.casefold() for t in sources(select)}
    candidates = []
    for name, columns in schema.columns.items():
        for item in columns.values():
            col = item["name"]
            words = parts(col)
            shared = words & topic
            if not shared or words & {"id", "code", "key", "zip", "phone", "number"}:
                continue
            if not re.search(r"text|char|varchar|string|clob", item["type"], re.I):
                continue
            score = 3 * len(shared) - .5 * len(words - topic) + .35 * (name in tables)
            score += .2 * len(parts(schema.tables[name]) & topic)
            candidates.append((score, name, col))
    for _, table, field in sorted(candidates, key=lambda value: (-value[0], value[1], value[2]))[:3]:
        changed = select.copy()
        if not schema.attach(changed, table):
            continue
        alias = next(t.alias_or_name for t in sources(changed) if t.name.casefold() == table)
        changed.set("order", exp.Order(expressions=[exp.Ordered(this=column(field, alias), desc=False)]))
        yield Edit("alphabetical", changed, "sort by the named text attribute instead of its key", .91)
        return


def followups(tree, branch, schema, intent, request, example, siblings, diagnosis, ordered):
    if not isinstance(tree, exp.Select):
        return
    if not branch.feedback.get("ok"):
        yield from local_binding(tree, schema, intent, branch.feedback)
        return
    yield from alphabetical_field(tree, schema, intent, ordered)
    yield from entity_name_filter(tree, schema, intent, diagnosis)
    if diagnosis.matched:
        if diagnosis.empty:
            yield from explicit_absence(tree, schema, request)
        return
    yield from explicit_absence(tree, schema, request)
    yield from aggregate_filter(tree, schema, intent, example)
    yield from unneeded_filter(tree, schema, intent, diagnosis)
    if diagnosis.rows in {"same", "more"} and request.quoted:
        for sibling in siblings:
            yield from grounded_filter(tree, sibling, schema, request, intent)
