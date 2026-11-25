"""Rebuild a small relational plan from current-schema bindings."""
from dataclasses import dataclass
import re

from sqlglot import exp

from .obligations import Request
from .skills import Edit, conditions, select_headers
from .structure import column, identifier, norm, parts, sources


WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10}
CUE = {
    "threshold": r"\b(?:at least|at most|more than|fewer than|exactly|greater than|less than)\b",
    "group": r"\b(?:each|per|for every|grouped)\b",
    "rank": r"\b(?:most|least|fewest|greatest|maximum|minimum|top|largest|smallest|highest|lowest)\b",
    "count": r"\b(?:how many|number of|count|times|frequen\w*|common)\b",
    "both": r"\bboth\b",
    "union": r"\b(?:either|or)\b",
    "absence": r"\b(?:never|without|except|not|no|don.t|doesn.t)\b",
    "distinct": r"\b(?:distinct|different|unique)\b",
}


def cues(question):
    rank_text = re.sub(r"\bat (?:least|most)\b", "", question, flags=re.I)
    return frozenset(kind for kind, pattern in CUE.items()
                     if re.search(pattern, rank_text if kind == "rank" else question, re.I))


@dataclass
class Frame:
    select: exp.Select
    owners: tuple
    population: str


class Binder:
    """Resolve relation and column owners without consulting any database rows."""
    def __init__(self, query, schema, intent):
        self.schema = schema
        self.intent = intent
        self.present = {table.alias_or_name.casefold(): table.name.casefold() for table in sources(query)}

    def owner(self, field, preferred=None):
        name = field.name.casefold()
        hint = self.present.get(field.table.casefold(), field.table.casefold())
        if name in self.schema.columns.get(hint, {}):
            return hint, self.schema.columns[hint][name]["name"]
        options = []
        for relation, columns in self.schema.columns.items():
            for meta in columns.values():
                candidate = meta["name"]
                if norm(candidate) != norm(name):
                    continue
                table_words = parts(self.schema.tables[relation])
                score = (2.2 * (relation == preferred) + 1.3 * (relation in self.present.values())
                         + 1.2 * (relation == hint) + .36 * len(table_words & self.intent.words))
                options.append((score, relation, candidate))
        if not options:
            return None
        _, table, col = sorted(options, key=lambda x: (-x[0], x[1], x[2]))[0]
        return table, col

    def bind(self, expression, preferred=None):
        if expression.find(exp.Subquery, exp.Select):
            return None
        clone = expression.copy()
        tables = []
        for field in list(clone.find_all(exp.Column)):
            binding = self.owner(field, preferred)
            if binding is None:
                if field.name == "*" or isinstance(field.this, exp.Star):
                    continue
                return None
            table, name = binding
            field.set("this", identifier(name))
            field.set("table", identifier(self.schema.tables[table]))
            tables.append(table)
        return clone, tables


def build_frame(query, schema, intent, example):
    if not isinstance(query, exp.Select) or query.args.get("with") or query.args.get("with_"):
        return None
    if any(not isinstance(join.this, exp.Table) for join in query.args.get("joins", [])):
        return None
    normalized = select_headers(query, example["columns"], schema, intent)
    if normalized is None:
        return None
    binder = Binder(normalized, schema, intent)
    studied = list(dict.fromkeys(binder.present.values()))
    outputs, owners = [], []
    for entry in normalized.expressions:
        bound = binder.bind(entry.unalias(), owners[0] if owners else None)
        if bound is None:
            return None
        outputs.append(bound[0])
        owners.extend(bound[1])
    predicates, criteria = [], []
    where = normalized.args.get("where")
    if where:
        for predicate in conditions(normalized):
            bound = binder.bind(predicate, owners[0] if owners else None)
            if bound is None:
                return None
            predicates.append(bound[0])
            criteria.extend(bound[1])
    main = (owners or criteria or studied)
    if not main or main[0] not in schema.tables:
        return None
    population = main[0]
    rebuilt = exp.Select(expressions=outputs).from_(exp.Table(this=identifier(schema.tables[population])))
    for relation in dict.fromkeys(owners + criteria):
        if not schema.attach(rebuilt, relation):
            return None
    if predicates:
        rebuilt.where(exp.and_(*predicates), copy=False)
    if normalized.args.get("distinct"):
        rebuilt.set("distinct", exp.Distinct())
    return Frame(rebuilt, tuple(dict.fromkeys(owners)), population)


def join_plan(query, frame, schema, intent):
    if isinstance(query, exp.Select) and any(query.args.get(k) for k in ("group", "having", "limit", "order")):
        return
    if query.find(exp.Subquery, exp.SetOperation) or not query.args.get("joins"):
        return
    rebuilt = frame.select.copy()
    if len(rebuilt.args.get("joins", [])) == 0 and len(frame.owners) <= 1:
        # If the original needs a relation for membership, keep the mentioned relation.
        related = [t.name.casefold() for t in sources(query) if t.name.casefold() != frame.population
                   and parts(t.name) and parts(t.name) <= intent.words]
        if related and not schema.attach(rebuilt, related[0]):
            return
    yield Edit("plan_join", rebuilt, "rebind output and filter owners through their FK paths", .86)


def rank_limit(question):
    match = re.search(r"\b(?:top|bottom|first|last)\s+(\d+|one|two|three|four|five)\b", question)
    if match:
        return int(match[1]) if match[1].isdigit() else WORDS[match[1]]
    return 1


def related_fact(frame, query, schema, intent):
    population = frame.population
    present = {table.name.casefold() for table in sources(frame.select)}
    previous = {table.name.casefold() for table in sources(query)}
    ranked = []
    for relation, table in schema.tables.items():
        if relation == population:
            continue
        route = schema.path([population], relation)
        if not route:
            continue
        terms = parts(table)
        overlap = terms & intent.words
        if not overlap and relation not in previous:
            continue
        score = 1.5 * len(overlap) + .65 * (relation in previous)
        score += .5 * (relation in present) - .45 * len(route)
        ranked.append((score, relation))
    return [relation for _, relation in sorted(ranked, key=lambda x: (-x[0], x[1]))[:2]]


def group_plan(query, frame, schema, intent, tags, request):
    has_count = any(p.find(exp.Count) for p in frame.select.expressions)
    columns = [e.copy() for e in frame.select.expressions if not e.find(exp.AggFunc)]
    if not columns:
        return
    prior_group = query.args.get("group")
    rank = "rank" in tags and "count" in tags
    per_entity = "group" in tags and has_count
    bound_count = bool(request.boundary and ("threshold" in tags)
                       and (prior_group or query.find(exp.Count) or query.find(exp.Having)))
    if not (rank or per_entity or bound_count):
        return
    if len(columns) > 4:
        return
    variants = related_fact(frame, query, schema, intent)
    if not variants or frame.select.args.get("joins"):
        variants = [None] + variants[:1]
    for relation in variants[:2]:
        rebuilt = frame.select.copy()
        if relation and not schema.attach(rebuilt, relation):
            continue
        whole_entity = any(isinstance(e, exp.Column) and e.name.casefold() == (schema.primary(frame.population) or "").casefold()
                           for e in columns)
        main_pk = schema.primary(frame.population)
        main_name = schema.tables[frame.population]
        if main_pk and (whole_entity or (len(columns) > 1 and len(frame.owners) == 1)):
            keys = [column(main_pk, main_name)]
        else:
            keys = columns
        rebuilt.group_by(*[e.copy() for e in keys], copy=False)
        if bound_count:
            operator, value = request.boundary[0]
            rebuilt.having(operator(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(value)), copy=False)
        if rank:
            descending = not bool(re.search(r"\b(?:least|fewest|minimum|smallest|lowest|bottom)\b", intent.text))
            rebuilt.order_by(exp.Ordered(this=exp.Count(this=exp.Star()), desc=descending), copy=False)
            if not re.search(r"\b(?:ordered|sorted)\s+by\s+(?:the\s+)?number\b", intent.text):
                rebuilt.limit(rank_limit(intent.text), copy=False)
        elif query.args.get("order") and any(w in intent.words for w in ("alphabetical", "ascending", "descending")):
            rebuilt.order_by(exp.Ordered(this=columns[0].copy(), desc=intent.descending is True), copy=False)
        yield Edit("plan_group", rebuilt, "rebuild grouping and count at the requested grain", .89)


def simple_set_plan(query, frame, tags):
    if not tags & {"both", "union"} or query.args.get("group") or query.args.get("limit"):
        return
    if frame.select.find(exp.AggFunc) or len(frame.select.expressions) > 5:
        return
    where = frame.select.args.get("where")
    if where is None:
        return
    rule = where.this
    if isinstance(rule, exp.In) and len(rule.expressions) == 2 and rule.args.get("query") is None:
        alternatives = [exp.EQ(this=rule.this.copy(), expression=value.copy()) for value in rule.expressions]
    elif isinstance(rule, (exp.And, exp.Or)) and all(isinstance(p, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE))
                                                       for p in (rule.this, rule.expression)):
        alternatives = [rule.this.copy(), rule.expression.copy()]
    else:
        return
    if "both" in tags:
        node = exp.Intersect
    elif "distinct" in tags and "union" in tags:
        node = exp.Union
    else:
        return
    sides = [frame.select.copy().where(predicate, append=False, copy=False) for predicate in alternatives]
    yield Edit("plan_set", node(this=sides[0], expression=sides[1]),
               "require the specified memberships over the same output entity", .91)


def absence_plan(query, frame, schema, intent, request):
    if not request.absence or query.args.get("group") or query.args.get("limit") or query.find(exp.Subquery):
        return
    if frame.select.find(exp.AggFunc) or any(owner != frame.population for owner in frame.owners):
        return
    matching = frame.select.copy()
    if not matching.args.get("joins"):
        facts = related_fact(frame, query, schema, intent)
        if not facts or not schema.attach(matching, facts[0]):
            return
    query_where = matching.args.get("where")
    if query_where:
        for node in list(query_where.find_all(exp.NEQ)):
            node.replace(exp.EQ(this=node.this.copy(), expression=node.expression.copy()))
        for node in list(query_where.find_all(exp.Not)):
            if isinstance(node.this, (exp.In, exp.Like)):
                node.replace(node.this.copy())
    total = exp.Select(expressions=[value.copy() for value in matching.expressions]).from_(
        exp.Table(this=identifier(schema.tables[frame.population])))
    yield Edit("plan_absence", exp.Except(this=total, expression=matching),
               "subtract the related entities from the requested population", .9)


def plans(query, schema, intent, request, example, rules):
    frame = build_frame(query, schema, intent, example)
    if frame is None:
        return
    tags = cues(intent.text)
    if rules.get("join", {}).get("examples", 0):
        yield from join_plan(query, frame, schema, intent)
    if rules.get("group", {}).get("examples", 0):
        yield from group_plan(query, frame, schema, intent, tags, request)
    if rules.get("set", {}).get("examples", 0):
        yield from simple_set_plan(query, frame, tags)
        yield from absence_plan(query, frame, schema, intent, request)
