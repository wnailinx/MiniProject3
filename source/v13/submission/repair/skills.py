"""Parameterized AST actions. Bindings come from the current task's schema."""
from dataclasses import dataclass
import re

from sqlglot import exp

from .structure import column, header_agreement, header_label, identifier, label, local, norm, parse, parts, sources, sql_text


@dataclass(frozen=True)
class Edit:
    kind: str
    tree: exp.Query
    reason: str
    confidence: float


SKILLS = {
    "projection": ["Output arity/order or headers disagree", "Bind public output expressions to in-scope columns; attach via FK if needed", "Result headers may be aliases or share names"],
    "binding": ["Missing table or column", "Resolve names in scope or connect a uniquely owning table using a declared FK path", "Similar names can refer to unrelated attributes"],
    "literal": ["String literal is inconsistent with a schema binding", "Bind literal to a compatible text field or split a full name", "An empty result is weak evidence"],
    "group": ["Grouped count, sum, or frequency requested", "Use projected keys and aggregate the appropriate record set", "Groups may require entity keys or a more specific count"],
    "boundary": ["The question explicitly gives a numeric boundary", "Change matching numeric comparison", "Equal literals can occur in unrelated predicates"],
    "distinct": ["Observed or possible duplicates", "Toggle result DISTINCT or count the entity key", "Some repetitions are intended"],
    "compact": ["Unused joined tables or gratuitous conditions", "Remove one unused leaf join or simplify clearly unnecessary constraints", "An apparently unused join can still filter rows"],
    "semi": ["Entity membership or negation across a join", "Convert a qualifying leaf join to a correlated EXISTS or NOT EXISTS", "An inner join may intentionally duplicate rows"],
    "order": ["Explicit ranking or sorting cue", "Change the existing ORDER BY or requested limit", "Ranking references can be ambiguous"],
    "transfer": ["A sibling branch has a relevant clause missing from this branch", "Transplant one compatible clause or projection between the same alias scopes", "Locally compatible clauses can express different intent"],
    "contrast": ["Negation, both, or relative comparison", "Repair a set operation or a scalar extremum based on explicit wording", "Cue words are only hypotheses"],
    "skeleton": ["Simple requested columns or statistic on a table", "Construct a small SELECT with current-schema bindings and a retained applicable predicate", "Dropping joins or clauses can lose constraints"],
    "multiplicity": ["COUNT arguments or group entity disagree with the request", "Choose row count when the request counts records; bind the grouped entity to a source key", "NULL and duplicate treatments depend on the requested entity"],
    "predicate": ["Explicit exclusion, string filter, or quantified comparison", "Bind operators, literal spelling, and correlated keys in the existing condition", "Case, nulls and quantifiers affect equivalence"],
    "relationship": ["Join equality disagrees with the schema", "Reconnect two scoped tables on a declared FK or short FK path", "Schemas can have multiple legitimate relationships"],
    "set_logic": ["Two values are joined by both, either, or an exclusion", "Build set operations over the same projected entity", "Distinct sets and row-level AND/OR can mean different things"],
    "single_relation": ["The requested projection and filters may live on one relation", "Bind a frame to a schema relation and recreate transferable predicates", "An unused-looking join may be part of the intended domain"],
    "value_repair": ["A public diagnostic found an alternative encoding", "Replace only the inspected literal in its checked parent", "A value matching the public instance may not match the intended predicate elsewhere"],
    "reshape": ["The expected result refers to a specific top, grouped or ordered output", "Shift existing aggregate/column expression into ORDER or add correlated HAVING", "Implicit order and tied ranks can differ"],
}


def select_headers(select, headers, schema, intent):
    """Use the public result's column contract as evidence, including expression labels."""
    changed = select.copy()
    projections = list(changed.expressions)
    used, bound = set(), []
    for index, header in enumerate(headers):
        expected = header_label(header)
        agreeing = [i for i, value in enumerate(projections) if i not in used and label(value) == expected]
        if agreeing:
            i = index if index in agreeing else agreeing[0]
            used.add(i)
            bound.append(projections[i].copy())
            continue
        tree = parse("SELECT " + header)
        if isinstance(tree, exp.Select) and len(tree.expressions) == 1 and re.fullmatch(
                r'[\w ."`\[\](),*+-]+', header):
            expression = tree.expressions[0].copy()
        elif re.fullmatch(r"[\w ]+", header):
            expression = column(header)
        else:
            return None
        if not isinstance(expression, (exp.Column, exp.AggFunc, exp.Binary)):
            return None
        for node in list(expression.find_all(exp.Column)):
            old = projections[index].unalias() if index < len(projections) else None
            alias = old.table if isinstance(old, exp.Column) else node.table
            options = schema.bindings(changed, node.name, intent, alias)
            if not options:
                return None
            choice = next((o for o in options if schema.attach(changed, o[1])), None)
            if choice is None:
                return None
            node.set("this", identifier(choice[3]))
            node.set("table", identifier(choice[2]))
        if index < len(projections) and projections[index].alias and isinstance(expression, exp.AggFunc):
            expression = exp.alias_(expression, projections[index].alias)
        bound.append(expression)
    if not bound:
        return None
    changed.set("expressions", bound)
    return changed


def projection(select, headers, schema, intent):
    agreement = header_agreement(select, headers)
    if agreement >= .99:
        return
    repaired = select_headers(select, headers, schema, intent)
    if repaired is not None:
        has_aggregate = any(e.find(exp.AggFunc) for e in repaired.expressions)
        scalars = [e.unalias().copy() for e in repaired.expressions if not e.find(exp.AggFunc)]
        if has_aggregate and scalars and not repaired.args.get("group"):
            repaired.group_by(*scalars, copy=False)
        yield Edit("projection", repaired, "public column contract", .93 if agreement < .65 else .76)


def binding(select, schema, intent, feedback):
    error = (feedback.get("error") or {}).get("message", "")
    table_error = re.search(r"no such table:\s*(.+)$", error, re.I)
    missing = re.search(r"(?:no such|ambiguous) column(?: name)?:\s*(.+)$", error, re.I)
    if table_error:
        raw = table_error[1].strip()
        resolved = schema.resolve_table(raw)
        if resolved:
            changed = select.copy()
            for table in local(changed, exp.Table):
                if table.name.casefold() == raw.casefold():
                    if not table.alias and resolved.casefold() != raw.casefold():
                        table.set("alias", exp.TableAlias(this=identifier(raw)))
                    table.set("this", identifier(resolved))
                    yield Edit("binding", changed, "resolve table spelling", .88)
                    break
    if not missing:
        return
    raw = missing[1].strip().strip('"')
    qualifier, _, name = raw.rpartition(".")
    matches = schema.bindings(select, name, intent, qualifier, fuzzy=True)
    for _, table, alias, actual in matches[:2]:
        changed = select.copy()
        if not schema.attach(changed, table):
            continue
        for node in local(changed, exp.Column):
            if node.name.casefold() == name.casefold() and (not qualifier or node.table.lower() == qualifier.lower()):
                node.set("table", identifier(alias))
                node.set("this", identifier(actual))
            elif not node.table:
                orig = schema.bindings(select, node.name, intent)
                scope = {t.name.casefold() for t in sources(select)}
                present = [item for item in orig if item[1] in scope]
                if len(present) == 1:
                    node.set("table", identifier(present[0][2]))
        yield Edit("binding", changed, "repair reported column binding", .90 if norm(actual) == norm(name) else .61)


def conditions(select):
    where = select.args.get("where")
    return [] if where is None else list(where.this.flatten()) if isinstance(where.this, exp.And) else [where.this]


def literal(select, schema, intent, feedback):
    for position, predicate in enumerate(conditions(select)):
        if not isinstance(predicate, (exp.EQ, exp.NEQ, exp.Like)) or not isinstance(predicate.this, exp.Column):
            continue
        value = predicate.expression
        if not isinstance(value, exp.Literal) or not value.is_string:
            continue
        field = predicate.this
        text = value.this
        for table in sources(select):
            cols = schema.columns.get(table.name.casefold(), {})
            first = next((c["name"] for c in cols.values() if norm(c["name"]) in ("firstname", "fname", "givenname")), None)
            last = next((c["name"] for c in cols.values() if norm(c["name"]) in ("lastname", "lname", "surname")), None)
            if first and last and (not field.table or field.table.casefold() == table.alias_or_name.casefold()):
                tokens = text.split()
                if len(tokens) == 2 and norm(field.name) in {"name", norm(first), norm(last)} and all(t.isalpha() for t in tokens):
                    changed = select.copy()
                    node = conditions(changed)[position]
                    node.replace(exp.and_(
                        exp.EQ(this=column(first, table.alias_or_name), expression=exp.Literal.string(tokens[0])),
                        exp.EQ(this=column(last, table.alias_or_name), expression=exp.Literal.string(tokens[1]))))
                    yield Edit("literal", changed, "split full name into schema fields", .79)
                    break
        if feedback.get("row_count") != 0 or not text.strip():
            continue
        options = []
        for table in sources(select):
            for candidate in schema.columns.get(table.name.casefold(), {}).values():
                name = candidate["name"]
                role = parts(name)
                if name.casefold() == field.name.casefold() or not role & {"name", "description", "title", "method", "code", "type"}:
                    continue
                overlap = len(role & parts(field.name))
                relevance = len(role & intent.words)
                if not overlap and not relevance:
                    continue
                score = .2 * overlap + .16 * relevance + .12 * (table.alias_or_name == field.table)
                options.append((score, table.alias_or_name, name))
        for score, alias, name in sorted(options, reverse=True)[:2]:
            changed = select.copy()
            conditions(changed)[position].set("this", column(name, alias))
            yield Edit("literal", changed, "rebind literal to a related field", min(.75, .42 + score))


def boundary(select, intent):
    phrases = [(r"at least|no fewer than", exp.GTE), (r"at most|no more than", exp.LTE),
               (r"more than|greater than|above|over", exp.GT),
               (r"less than|fewer than|below|under", exp.LT), (r"exactly", exp.EQ)]
    number_words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
    number_pattern = r"(\d+(?:\.\d+)?|one|two|three|four|five)\b"
    for pattern, needed in phrases:
        match = re.search(r"\b(?:" + pattern + r")\s+" + number_pattern, intent.text)
        if match is None:
            continue
        number = number_words.get(match[1], match[1])
        changed = select.copy()
        for node in changed.walk():
            if not isinstance(node, (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ)) or type(node) is needed:
                continue
            if not isinstance(node.expression, exp.Literal) or node.expression.is_string or node.expression.this != number:
                continue
            if isinstance(node, exp.EQ) and needed is not exp.EQ:
                continue
            node.replace(needed(this=node.this.copy(), expression=node.expression.copy()))
            yield Edit("boundary", changed, "explicit numeric comparison", .86)
            break
        break


def grouping(select, schema, intent, headers):
    if not (intent.frequent or intent.grouped or (intent.count and "group" in intent.words)):
        return
    changed = select_headers(select, headers, schema, intent)
    if changed is None:
        return
    normal = [e.unalias().copy() for e in changed.expressions if not e.find(exp.AggFunc)]
    if not normal:
        return
    if intent.frequent and select.args.get("order") is not None:
        existing_order = select.args["order"].expressions[0].this
        if isinstance(existing_order, exp.Column) and existing_order.name.lower() in {"count", "frequency"}:
            return
    if not changed.args.get("group"):
        changed.group_by(*normal, copy=False)
    if intent.frequent:
        count = exp.Count(this=exp.Star())
        if not select.args.get("group"):
            changed.set("having", None)
        changed.set("order", exp.Order(expressions=[exp.Ordered(this=count, desc=intent.descending is not False)]))
        if not changed.args.get("limit"):
            changed.limit(1, copy=False)
    elif not any(e.find(exp.AggFunc) for e in changed.expressions):
        return
    yield Edit("group", changed, "frequency or per-entity grouping", .88 if intent.frequent else .77)


def grouped_threshold(select, schema, intent, headers):
    if select.args.get("group") or not select.args.get("where") or intent.count:
        return
    pattern = r"\b(at least|at most|more than|fewer than|exactly)\s+(\d+|one|two|three|four|five)"
    threshold = re.search(pattern, intent.text)
    if threshold is None or len(headers) != 1 or len(sources(select)) != 1:
        return
    condition = select.args["where"]
    if not condition.find(exp.Subquery) or not condition.find(exp.Having):
        return
    copy = select_headers(select, headers, schema, intent)
    if copy is None or copy.expressions[0].find(exp.AggFunc):
        return
    word_numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    operand = word_numbers.get(threshold[2], threshold[2])
    compare = {"at least": exp.GTE, "at most": exp.LTE, "more than": exp.GT,
               "fewer than": exp.LT, "exactly": exp.EQ}[threshold[1]]
    copy.set("where", None)
    copy.set("distinct", None)
    copy.group_by(copy.expressions[0].unalias().copy(), copy=False)
    copy.having(compare(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(operand)), copy=False)
    yield Edit("group", copy, "group directly by the requested property", .84)


def multiplicity(select, intent, headers):
    if not (intent.count or intent.frequent or select.args.get("having")):
        return
    text = intent.text
    count_star = any(header_label(header) == "count" for header in headers)
    changed = select.copy()
    replacements = 0
    for node in local(changed, exp.Count):
        value = node.this
        if isinstance(value, exp.Star):
            continue
        if isinstance(value, exp.Distinct) and intent.distinct:
            continue
        if count_star or intent.frequent or node.find_ancestor(exp.Having) is not None:
            node.set("this", exp.Star())
            replacements += 1
    if replacements:
        yield Edit("multiplicity", changed, "count records rather than arbitrary values", .83 if count_star else .60)
    if select.args.get("having") or select.args.get("where") is None:
        return
    numwords = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    match = re.search(r"\b(at least|more than|at most|less than|exactly)\s+(\d+|one|two|three|four|five)\b", text)
    if not match or not select.find(exp.Group):
        return
    if len(select.expressions) != 1 or select.expressions[0].find(exp.AggFunc):
        return
    number = numwords[match[2]] if match[2] in numwords else int(match[2])
    compare = {"at least": exp.GTE, "more than": exp.GT, "at most": exp.LTE,
               "less than": exp.LT, "exactly": exp.EQ}[match[1]]
    changed = select.copy()
    changed.having(compare(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(number)), copy=False)
    yield Edit("multiplicity", changed, "apply explicit threshold to a grouped record count", .73)


def predicate_edits(select, schema, intent, feedback):
    exact_strings = re.findall(r"""(?<!\w)['"]([^\n'"]{2,80})['"](?!\w)""", intent.text)
    for node in select.find_all(exp.Literal):
        if not node.is_string:
            continue
        value = node.this
        if not value.strip() or len(value) > 100:
            continue
        parent = node.parent
        if isinstance(parent, exp.Like) and value.startswith("%") and value.endswith("%") and value.strip("%").casefold() in exact_strings:
            changed = select.copy()
            related = next((n for n in changed.find_all(exp.Like) if n == parent), None)
            if related is not None:
                related.replace(exp.EQ(this=related.this.copy(), expression=exp.Literal.string(value.strip("%"))))
                yield Edit("predicate", changed, "quoted value is an exact comparison", .78)
        if not isinstance(parent, (exp.EQ, exp.NEQ, exp.In)) or (
                feedback.get("row_count", 0) > 1 and feedback.get("matches_example")):
            continue
        variants = []
        if value.casefold() in intent.text and value.casefold() != value:
            variants.append((value.casefold(), .63))
        if value.casefold() in intent.text and value.title() != value:
            variants.append((value.title(), .61))
        if value.isdigit() and re.search(r"\b(minutes?|duration)\b", intent.text):
            variants.insert(0, (value + ":00", .81))
        if re.search(r"\bteam\s+" + re.escape(value.casefold()) + r"\b", intent.text):
            variants.insert(0, ("Team " + value, .84))
        for replacement, confidence in variants[:2]:
            changed = select.copy()
            for literal_node in changed.find_all(exp.Literal):
                if literal_node.is_string and literal_node.this == value:
                    literal_node.set("this", replacement)
            yield Edit("predicate", changed, "repair the literal representation", confidence)
        if variants:
            break
    if re.search(r"\b(older|younger|higher|lower|greater|less|more)\s+than\b", intent.text):
        wants_average = "average" in intent.words
        wants_extreme = bool(re.search(r"\b(all|every|any)\b", intent.text))
        if wants_average or wants_extreme:
            for subquery in select.find_all(exp.Subquery):
                comparison = subquery.parent
                if not isinstance(comparison, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
                    continue
                query = subquery.this
                if not isinstance(query, exp.Select) or len(query.expressions) != 1 or query.expressions[0].find(exp.AggFunc):
                    continue
                changed = select.copy()
                target = next(q for q in changed.find_all(exp.Subquery) if q == subquery)
                greater = isinstance(comparison, (exp.GT, exp.GTE))
                extreme = exp.Max if greater != bool(re.search(r"\bany\b", intent.text)) else exp.Min
                fn = exp.Avg if wants_average else extreme
                target.this.set("expressions", [fn(this=query.expressions[0].unalias().copy())])
                yield Edit("predicate", changed, "quantified scalar comparison", .86)
                break
        if wants_average and not select.find(exp.Subquery) and len(sources(select)) == 1:
            table = sources(select)[0]
            target = next((c["name"] for c in schema.columns.get(table.name.casefold(), {}).values()
                           if parts(c["name"]) & {"age", "aged"} and re.search(r"\b(older|younger)\b", intent.text)), None)
            if target:
                changed = select.copy()
                sub = exp.select(exp.Avg(this=column(target))).from_(exp.Table(this=table.this.copy()))
                compare = exp.LT if "younger" in intent.words else exp.GT
                changed.where(compare(this=column(target, table.alias_or_name), expression=exp.Subquery(this=sub)), copy=False)
                yield Edit("predicate", changed, "compare against the table average", .84)
    if not re.search(r"\b(never|without|do not|does not|did not|don't)\b", intent.text):
        return
    predicates = conditions(select)
    if len(predicates) == 1 and not select.args.get("group") and select.args.get("joins") is None:
        predicate = predicates[0]
        if not predicate.find(exp.Subquery) and isinstance(predicate, (exp.EQ, exp.In)):
            plain = select.copy()
            plain.set("where", None)
            plain.set("order", None)
            plain.set("limit", None)
            yield Edit("predicate", exp.Except(this=plain, expression=select.copy()),
                       "exclude the set of matching entities", .78)
    for index, join, alias, belongs in leaves(select, schema):
        if len(select.args["joins"]) != 1 or select.find(exp.Group):
            continue
        if any(belongs(c, alias) for e in select.expressions for c in e.find_all(exp.Column)):
            continue
        if len(predicates) != 1:
            continue
        predicate = predicates[0]
        if isinstance(predicate, exp.Not):
            predicate = predicate.this
        elif isinstance(predicate, exp.NEQ):
            predicate = exp.EQ(this=predicate.this.copy(), expression=predicate.expression.copy())
        if not isinstance(predicate, (exp.EQ, exp.In, exp.Like, exp.GT, exp.GTE)):
            continue
        if not any(belongs(c, alias) for c in predicate.find_all(exp.Column)):
            continue
        corrected = select.copy()
        corrected.set("joins", None)
        restriction = exp.and_(join.args["on"].copy(), predicate.copy())
        inside = exp.select("1").from_(join.this.copy()).where(restriction)
        corrected.where(exp.Not(this=exp.Exists(this=inside)), append=False, copy=False)
        yield Edit("predicate", corrected, "absence of a related matching record", .88)
        break


def relationships(select, schema):
    tables = sources(select)
    if len(tables) < 2:
        return
    scope = {table.alias_or_name.casefold(): table.name.casefold() for table in tables}
    for index, join in enumerate(select.args.get("joins", [])):
        condition = join.args.get("on")
        if not isinstance(condition, exp.EQ) or not all(isinstance(x, exp.Column) for x in (condition.this, condition.expression)):
            continue
        left, right = condition.this, condition.expression
        source, target = scope.get(left.table.casefold()), scope.get(right.table.casefold())
        if source is None or target is None or source == target:
            continue
        path = schema.path([source], target)
        if not path or (len(path) == 1 and any(
                pair[0] == target and pair[1].casefold() == left.name.casefold() and pair[2].casefold() == right.name.casefold()
                for pair in schema.edges[source])):
            continue
        # Inserting a bridge requires the source to precede its target in FROM.
        aliases = {name: table.alias_or_name for table in tables for name in [table.name.casefold()]}
        if join.this.alias_or_name.casefold() == left.table.casefold():
            path = schema.path([target], source)
        if not path:
            continue
        changed = select.copy()
        generated = []
        for a, b, ca, cb in path:
            alias = aliases.get(b, schema.tables[b])
            table = (join.this.copy() if b == scope[join.this.alias_or_name.casefold()]
                     else exp.Table(this=identifier(schema.tables[b])))
            if b in aliases and table != join.this:
                generated = []
                break
            generated.append(exp.Join(this=table, on=exp.EQ(
                this=column(ca, aliases.get(a, schema.tables[a])), expression=column(cb, alias))))
        if generated:
            changed.set("joins", changed.args["joins"][:index] + generated + changed.args["joins"][index + 1:])
            yield Edit("relationship", changed, "restore a declared join relationship", .81)
            break


def duplicates(select, schema, intent, diagnosis, probe):
    if select.args.get("distinct") or diagnosis.rows == "more" or intent.distinct or diagnosis.join or diagnosis.empty:
        if not any(e.find(exp.AggFunc) for e in select.expressions):
            changed = select.copy()
            changed.set("distinct", None if changed.args.get("distinct") else exp.Distinct())
            yield Edit("distinct", changed, "output multiplicity", .73 if intent.distinct or diagnosis.rows == "more" else .46)
    if not diagnosis.join:
        return
    for selected in select.expressions:
        count = selected.find(exp.Count)
        if count is None or isinstance(count.this, exp.Distinct):
            continue
        table = next((t for t in sources(select) if isinstance(count.this, exp.Column)
                      and count.this.table.casefold() == t.alias_or_name.casefold()), None)
        if table is None:
            table = next(iter(sources(select)), None)
        primary = schema.primary(table.name.casefold()) if table is not None else None
        if primary:
            changed = select.copy()
            changed.find(exp.Count).set("this", exp.Distinct(expressions=[column(primary, table.alias_or_name)]))
            observed = probe is not None and probe[0] > probe[1]
            yield Edit("distinct", changed, "count distinct entity key", .87 if observed or intent.distinct else .54)
        break


def order(select, intent, ordered):
    current = select.args.get("order")
    if current is not None and intent.descending is not None and not intent.frequent:
        first = current.expressions[0]
        if bool(first.args.get("desc")) != intent.descending:
            changed = select.copy()
            changed.args["order"].expressions[0].set("desc", intent.descending)
            yield Edit("order", changed, "explicit sort direction", .78)
    size = re.search(r"\b(?:top|first|bottom)\s+(\d+)\b", intent.text)
    if size and (current is not None or ordered):
        changed = select.copy()
        changed.limit(int(size[1]), copy=False)
        yield Edit("order", changed, "explicit result limit", .83)
    if current is None and ordered and intent.descending is not None:
        expressions = [p.unalias().copy() for p in select.expressions if isinstance(p.unalias(), exp.Column)]
        mentioned = [e for e in expressions if parts(e.name) & intent.words]
        if mentioned or expressions:
            changed = select.copy()
            chosen = (mentioned or expressions)[0]
            changed.set("order", exp.Order(expressions=[exp.Ordered(this=chosen, desc=intent.descending)]))
            yield Edit("order", changed, "restore requested result ordering", .72)


def leaves(select, schema):
    tables = sources(select)
    current = {t.alias_or_name.casefold(): t for t in tables}
    ownership = {}
    for table in tables:
        for name in schema.columns.get(table.name.casefold(), {}):
            ownership.setdefault(name, []).append(table.alias_or_name.casefold())
    def belongs(node, alias):
        return node.table.casefold() == alias if node.table else alias in ownership.get(node.name.casefold(), [])
    joins = select.args.get("joins", ())
    for index in range(len(joins) - 1, -1, -1):
        join = joins[index]
        if not isinstance(join.this, exp.Table) or join.args.get("on") is None:
            continue
        alias = join.this.alias_or_name.casefold()
        if alias not in current or any(belongs(col, alias) for j in joins[index + 1:]
                                      for col in j.find_all(exp.Column)):
            continue
        yield index, join, alias, belongs


def compact_and_semi(select, schema, intent, diagnosis):
    for index, join, alias, belongs in leaves(select, schema):
        visible = [node for name in ("expressions", "order", "group", "having")
                   for value in ([select.args[name]] if name != "expressions" and select.args.get(name) else
                                 select.expressions if name == "expressions" else [])
                   for node in value.find_all(exp.Column)]
        if any(belongs(col, alias) for col in visible):
            continue
        predicates = conditions(select)
        inner = [i for i, item in enumerate(predicates) if any(belongs(c, alias) for c in item.find_all(exp.Column))]
        changed = select.copy()
        changed.set("joins", [j.copy() for i, j in enumerate(select.args["joins"]) if i != index])
        if not inner:
            yield Edit("compact", changed.copy(), "remove unused leaf join", .66)
        if diagnosis.join and (inner or diagnosis.rows == "more" or intent.negative or intent.distinct):
            restriction = exp.and_(join.args["on"].copy(), *(predicates[i].copy() for i in inner))
            subquery = exp.select("1").from_(join.this.copy()).where(restriction)
            exists = exp.Exists(this=subquery)
            negative = intent.negative and not diagnosis.negative and bool(re.search(r"\b(never|without|do not|does not|did not|don't|no)\b", intent.text))
            keep = [item.copy() for i, item in enumerate(predicates) if i not in inner]
            changed.where(exp.and_(*keep, exp.Not(this=exists) if negative else exists), append=False, copy=False)
            yield Edit("semi", changed, "bind membership to one entity", .82 if negative else .63)
        break


def contrast(tree, intent):
    if isinstance(tree, exp.SetOperation):
        if re.search(r"\bboth\b", intent.text) and not isinstance(tree, exp.Intersect):
            yield Edit("contrast", exp.Intersect(this=tree.this.copy(), expression=tree.expression.copy()),
                       "both requires set intersection", .85)
        if re.search(r"\b(except|but not)\b", intent.text) and not isinstance(tree, exp.Except):
            yield Edit("contrast", exp.Except(this=tree.this.copy(), expression=tree.expression.copy()),
                       "except requires set exclusion", .85)
    if not isinstance(tree, exp.Select):
        return
    if re.search(r"\b(after|later|larger|higher|older|more|greater)\b.*\bany\b", intent.text):
        changed = tree.copy()
        for node in changed.find_all(exp.Max):
            if node.find_ancestor(exp.Subquery) is not None:
                node.replace(exp.Min(this=node.this.copy()))
                yield Edit("contrast", changed, "any calls for the comparison against the weakest member", .78)
                break


def set_logic(select, intent):
    if select.find(exp.AggFunc) or select.args.get("order") or select.args.get("limit"):
        return
    where = select.args.get("where")
    if where is None:
        return
    predicate = where.this
    if isinstance(predicate, exp.In) and predicate.args.get("query") is None and len(predicate.expressions) == 2:
        if re.search(r"\bboth\b", intent.text):
            members = [select.copy().where(exp.EQ(this=predicate.this.copy(), expression=value.copy()),
                                           append=False) for value in predicate.expressions]
            yield Edit("set_logic", exp.Intersect(this=members[0], expression=members[1]),
                       "both explicit values must associate with the same projected entity", .87)
    if isinstance(predicate, exp.And):
        values = [part for part in predicate.flatten() if isinstance(part, exp.In) and len(part.expressions) == 2]
        if len(values) == 1 and re.search(r"\b(either|or)\b.*\b(but|and)\b", intent.text):
            membership = values[0]
            sides = [exp.EQ(this=membership.this.copy(), expression=value.copy()) for value in membership.expressions]
            rest = [part.copy() for part in predicate.flatten() if part is not membership]
            changed = select.copy()
            changed.where(exp.or_(sides[0], exp.and_(sides[1], *rest)), append=False, copy=False)
            yield Edit("set_logic", changed, "apply the exceptional extra condition to one alternative", .81)
    if (not select.args.get("joins") or not re.search(r"\b(never|without|not living|do not|does not|don't)\b", intent.text)
            or not isinstance(predicate, (exp.NEQ, exp.Not))):
        return
    tables = sources(select)
    if len(tables) != 1 + len(select.args["joins"]):
        return
    main = tables[0]
    projected = [col for expression in select.expressions for col in expression.find_all(exp.Column)]
    if not projected or any(col.table.casefold() != main.alias_or_name.casefold() for col in projected):
        return
    included = (predicate.this.copy() if isinstance(predicate, exp.Not) else
                exp.EQ(this=predicate.this.copy(), expression=predicate.expression.copy()))
    excluded = select.copy()
    excluded.where(included, append=False, copy=False)
    population = select.copy()
    population.set("joins", None)
    population.set("where", None)
    yield Edit("set_logic", exp.Except(this=population, expression=excluded),
               "exclude matching projected entities across the join path", .86)


def reshape(select, schema, intent, example, feedback, ordered):
    question = intent.text
    current_order = select.args.get("order")
    current_group = select.args.get("group")
    keys = [p.unalias().copy() for p in select.expressions if isinstance(p.unalias(), exp.Column)]
    if ((current_group or intent.frequent) and keys and (intent.frequent or
            re.search(r"\b(number of|count of)\b", question) and re.search(r"\b(maximum|most|fewest|least)\b", question))
            and not (current_order and current_order.find(exp.Count))):
        changed = select_headers(select, example["columns"], schema, intent)
        if changed is not None:
            if not current_group:
                changed.group_by(*keys, copy=False)
            if current_group and select.args.get("having") and select.args["having"].find(exp.Subquery):
                changed.set("having", None)
            changed.set("order", exp.Order(expressions=[exp.Ordered(
                this=exp.Count(this=exp.Star()), desc=intent.descending is not False)]))
            if not changed.args.get("limit") and not re.search(r"\b(order|sort)\w* by\b", question):
                changed.limit(1, copy=False)
            yield Edit("reshape", changed, "order grouped records by their count", .88)
    if not (ordered and feedback.get("ok") and not feedback.get("matches_example") and keys
            and example["row_count"] == feedback["row_count"] and example["row_count"] >= 2):
        return
    header = [label(key) for key in keys]
    observed = feedback.get("rows", [])
    sample = example.get("rows", [])
    if not observed or not sample or sorted(map(str, observed)) != sorted(map(str, sample)):
        return
    for index, expected_label in enumerate(map(header_label, example["columns"])):
        if expected_label not in header:
            continue
        values = [row[index] for row in sample]
        if any(value is None or not isinstance(value, type(values[0])) for value in values) or len(set(values)) < 2:
            continue
        ascending = values == sorted(values)
        descending = values == sorted(values, reverse=True)
        if not ascending and not descending:
            continue
        changed = select.copy()
        changed.set("order", exp.Order(expressions=[exp.Ordered(this=keys[header.index(expected_label)].copy(),
                                                               desc=descending)]))
        yield Edit("reshape", changed, "public row order supports sorting this field", .83)
        break


def skeleton(select, schema, intent, headers):
    bound = select_headers(select, headers, schema, intent)
    if bound is None or not sources(bound):
        return
    projection_columns = list(col for proj in bound.expressions for col in proj.find_all(exp.Column))
    scope = sources(bound)
    possible = []
    for table in scope:
        columns = schema.columns.get(table.name.casefold(), {})
        if all(c.name.casefold() in columns for c in projection_columns):
            relevance = len(parts(table.name) & intent.words)
            possible.append((relevance, table))
    if not possible:
        return
    _, chosen = max(possible, key=lambda pair: pair[0])
    clean = exp.Select(expressions=[e.copy() for e in bound.expressions]).from_(chosen.copy())
    for col in local(clean, exp.Column):
        col.set("table", None)
    translated = []
    for predicate in conditions(select):
        cols = list(predicate.find_all(exp.Column))
        if not cols or predicate.find(exp.Subquery):
            continue
        if all(c.name.casefold() in schema.columns[chosen.name.casefold()] for c in cols):
            translated.append(predicate.transform(
                lambda node: column(node.name) if isinstance(node, exp.Column) else node))
    if translated:
        clean.where(exp.and_(*translated), copy=False)
    if intent.distinct and not clean.find(exp.AggFunc):
        clean.set("distinct", exp.Distinct())
    if intent.frequent:
        return
    # Do not drop grouping on a grouped task, or order on an explicit rank task.
    if intent.grouped or select.args.get("limit") or intent.descending is not None:
        return
    yield Edit("skeleton", clean, "simple single-table query from declared columns", .57)


def single_relation(select, schema, intent, headers):
    if select.find(exp.Exists) or select.args.get("having") or len(headers) > 5:
        return
    public_columns = select_headers(select, headers, schema, intent)
    if public_columns is None:
        return
    selectors = [node for e in public_columns.expressions for node in e.find_all(exp.Column)]
    predicates = conditions(select)
    if any(pred.find(exp.Subquery) or pred.find(exp.Select) for pred in predicates):
        return
    tables = sources(select)
    available = []
    for name, columns in schema.columns.items():
        if any(c.name.casefold() not in columns for c in selectors):
            continue
        label_tokens = parts(schema.tables[name])
        related = len(label_tokens & intent.words)
        if not related and not any(name == t.name.casefold() for t in tables):
            continue
        mapped = []
        for predicate in predicates:
            if not isinstance(predicate, (exp.EQ, exp.NEQ, exp.Like, exp.In, exp.Between, exp.GT, exp.GTE, exp.LT, exp.LTE)):
                break
            source_col = predicate.this
            if not isinstance(source_col, exp.Column):
                break
            lookup = source_col.name.casefold()
            if lookup not in columns:
                value = predicate.expression
                if not isinstance(value, exp.Literal):
                    break
                near = intent.text.find(value.this.casefold())
                context = parts(intent.text[max(0, near - 33):near + len(value.this) + 15]) if near >= 0 else set()
                allowed = [c["name"] for c in columns.values()
                           if parts(c["name"]) & context and (not value.is_string or
                              not re.search(r"int|real|float|numeric", c["type"], re.I))]
                if len(allowed) != 1:
                    break
                lookup = allowed[0].casefold()
            rewritten = predicate.copy()
            rewritten.set("this", column(columns[lookup]["name"]))
            mapped.append(rewritten)
        if len(mapped) != len(predicates):
            continue
        available.append((related, name, mapped))
    if not available:
        return
    for _, name, mapped in sorted(available, key=lambda item: (-item[0], item[1]))[:2]:
        simple = public_columns.copy()
        simple.set("from", None)
        simple.set("from_", None)
        simple.from_(exp.Table(this=identifier(schema.tables[name])), copy=False)
        simple.set("joins", None)
        simple.set("where", None)
        for node in local(simple, exp.Column):
            node.set("table", None)
        if any(node.name.casefold() not in schema.columns[name] and node.name not in
               {e.alias for e in simple.expressions} for node in local(simple, exp.Column)):
            continue
        if mapped:
            simple.where(exp.and_(*mapped), copy=False)
        yield Edit("single_relation", simple, "bind all projected fields and filters on one relation", .72)


def transfer(select, others, headers, schema):
    current = {t.name.casefold(): t.alias_or_name for t in sources(select)}
    for other in others[:4]:
        if not isinstance(other, exp.Select):
            continue
        sibling = {t.alias_or_name.casefold(): t.name.casefold() for t in sources(other)}
        if not current or not sibling or not set(current) & set(sibling.values()):
            continue
        def bound(values):
            adjusted = select.copy()
            copied = [value.copy() for value in values]
            for expression in copied:
                if expression.find(exp.Select):
                    return None, []
                for node in expression.find_all(exp.Column):
                    if not node.table:
                        options = [name for name in sibling.values() if node.name.casefold() in schema.columns.get(name, {})]
                        if len(options) != 1:
                            return None, []
                        table = options[0]
                    else:
                        table = sibling.get(node.table.casefold())
                    if not table or not schema.attach(adjusted, table):
                        return None, []
                    alias = next(t.alias_or_name for t in sources(adjusted) if t.name.casefold() == table)
                    node.set("table", identifier(alias))
            return adjusted, copied
        if header_agreement(other, headers) > header_agreement(select, headers):
            changed, replacement = bound(other.expressions)
            if changed is not None:
                changed.set("expressions", replacement)
                yield Edit("transfer", changed, "reuse better-aligned sibling projection", .69)
        for name in ("where", "having", "order"):
            old, new = select.args.get(name), other.args.get(name)
            if old is not None or new is None:
                continue
            if name == "having" and not select.args.get("group"):
                continue
            changed, replacement = bound([new])
            if changed is not None:
                changed.set(name, replacement[0])
                yield Edit("transfer", changed, "reuse a schema-bound missing sibling clause", .66)
                break


def edits(tree, schema, intent, example, feedback, diagnosis, ordered, others, probe):
    yield from contrast(tree, intent)
    if not isinstance(tree, exp.Select):
        return
    headers = example.get("columns", [])
    yield from projection(tree, headers, schema, intent)
    yield from binding(tree, schema, intent, feedback)
    yield from grouping(tree, schema, intent, headers)
    yield from grouped_threshold(tree, schema, intent, headers)
    yield from multiplicity(tree, intent, headers)
    yield from duplicates(tree, schema, intent, diagnosis, probe)
    yield from boundary(tree, intent)
    yield from order(tree, intent, ordered)
    yield from reshape(tree, schema, intent, example, feedback, ordered)
    yield from literal(tree, schema, intent, feedback)
    yield from predicate_edits(tree, schema, intent, feedback)
    yield from relationships(tree, schema)
    yield from set_logic(tree, intent)
    yield from compact_and_semi(tree, schema, intent, diagnosis)
    yield from transfer(tree, others, headers, schema)
    yield from skeleton(tree, schema, intent, headers)
    yield from single_relation(tree, schema, intent, headers)


def diagnostic_query(select, schema):
    if not isinstance(select, exp.Select) or select.args.get("group") or not select.args.get("joins"):
        return None
    tables = sources(select)
    if not tables:
        return None
    table = tables[0]
    primary = schema.primary(table.name.casefold())
    if not primary:
        return None
    query = select.copy()
    query.set("distinct", None)
    query.set("order", None)
    query.set("limit", None)
    query.set("offset", None)
    query.set("expressions", [exp.Count(this=exp.Star()),
                              exp.Count(this=exp.Distinct(expressions=[column(primary, table.alias_or_name)]))])
    return sql_text(query)


def literal_diagnostic(select, schema, intent):
    """Inspect one plausible text field in the public database for its encoding."""
    if not isinstance(select, exp.Select):
        return None
    tables = sources(select)
    for predicate in conditions(select):
        if not isinstance(predicate, (exp.EQ, exp.NEQ)) or not isinstance(predicate.this, exp.Column):
            continue
        value = predicate.expression
        if not isinstance(value, exp.Literal) or not value.is_string or not 2 <= len(value.this) <= 60:
            continue
        scoped = [t for t in tables if (not predicate.this.table or t.alias_or_name.casefold() == predicate.this.table.casefold())
                  and predicate.this.name.casefold() in schema.columns.get(t.name.casefold(), {})]
        if len(scoped) != 1 or value.this.casefold() not in intent.text:
            continue
        field = column(predicate.this.name)
        search = exp.Anonymous(this="INSTR", expressions=[exp.Lower(this=field.copy()),
                                                          exp.Literal.string(value.this.casefold())])
        filtered = exp.GT(this=search, expression=exp.Literal.number(0))
        query = exp.select(field.copy()).distinct().from_(exp.Table(this=scoped[0].this.copy()))
        query.where(filtered, copy=False).order_by(field.copy(), copy=False).limit(8, copy=False)
        return sql_text(query), value.this
    return None


def literal_from_probe(tree, original, values):
    if tree is None:
        return
    for value in values:
        changed = tree.copy()
        selected = next((node for node in changed.find_all(exp.Literal) if node.is_string and node.this == original), None)
        if selected is not None:
            selected.set("this", value)
            yield Edit("value_repair", changed, "public diagnostic found this literal representation", .95)
