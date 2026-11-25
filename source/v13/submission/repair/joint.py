"""Bounded AST compositions built from the current query and local cached hints."""
from collections import deque
from dataclasses import dataclass
import re

from sqlglot import exp

from .coverage_edits import grounded_predicate
from .obligations import predicate_nodes
from .plans import Binder, build_frame, cues, rank_limit
from .semantic import owner, scope
from .skills import conditions, relationships, select_headers
from .structure import column, header_agreement, identifier, invalid_bindings, key, local, parse, parts, sources, sql_text


@dataclass
class Construction:
    tree: exp.Query
    family: str
    source: str
    operations: tuple
    dependencies: tuple
    evidence: float


def routes(schema, start, target):
    if target in start:
        return [[]]
    queue = deque((name, [], {name}) for name in sorted(start))
    found = []
    while queue and len(found) < 2:
        current, path, seen = queue.popleft()
        if len(path) == 3:
            continue
        for adjacent, left, right in schema.edges.get(current, ()):
            if adjacent in seen:
                continue
            tail = path + [(current, adjacent, left, right)]
            if adjacent == target:
                found.append(tail)
            else:
                queue.append((adjacent, tail, seen | {adjacent}))
    return found


def reconstruct(query, schema, intent, example):
    frame = build_frame(query, schema, intent, example)
    if frame is None:
        return
    # Rebind every surviving clause into one consistent alias scope.
    binder = Binder(query, schema, intent)
    result = frame.select
    for name in ("group", "having", "order"):
        clause = query.args.get(name)
        if clause is not None:
            rebinding = binder.bind(clause, frame.population)
            if rebinding is None:
                return
            clause, owners = rebinding
            if not all(schema.attach(result, relation) for relation in owners):
                return
            result.set(name, clause)
    for name in ("limit", "offset"):
        if query.args.get(name):
            result.set(name, query.args[name].copy())
    yield result
    # Allow a second declared relationship path, preserving all expressions.
    needed = {field.table.casefold() for name in ("expressions", "where", "group", "having", "order")
              for item in (result.expressions if name == "expressions" else [result.args.get(name)])
              if isinstance(item, exp.Expression) for field in item.find_all(exp.Column) if field.table}
    if len(needed) == 2 and not result.find(exp.Subquery):
        main = sources(result)[0].name.casefold()
        other = next((table for table in needed if table != main), None)
        paths = routes(schema, {main}, other) if other else []
        for path in paths[1:2]:
            alternative = result.copy()
            alternative.set("joins", None)
            for a, b, left, right in path:
                alternative.join(exp.Table(this=identifier(schema.tables[b])), on=exp.EQ(
                    this=column(left, schema.tables[a]), expression=column(right, schema.tables[b])), copy=False)
            yield alternative


def filter_transplants(query, sibling, schema, intent, quoted):
    if not isinstance(sibling, exp.Select) or sibling.find(exp.Subquery) or query.find(exp.Subquery):
        return
    if not set(t.name.casefold() for t in sources(query)) & set(t.name.casefold() for t in sources(sibling)):
        return
    for predicate in conditions(sibling):
        literals = {value.this.strip("%").casefold() for value in predicate.find_all(exp.Literal) if value.is_string}
        if not literals or not literals & set(quoted):
            continue
        bound = Binder(sibling, schema, intent).bind(predicate)
        if bound is None:
            continue
        normalized, required = bound
        # Canonicalize this query first so the transplanted fields and aliases agree.
        base = next(reconstruct(query, schema, intent, {"columns": [p.sql() for p in query.expressions]}), None)
        if base is None or not all(schema.attach(base, table) for table in required):
            continue
        keep = []
        for old in conditions(base):
            old_values = {value.this.strip("%").casefold() for value in old.find_all(exp.Literal) if value.is_string}
            if not old_values & literals:
                keep.append(old.copy())
        base.where(exp.and_(*keep, normalized), append=False, copy=False)
        yield base


def joined_filters(tree, schema, intent, request, example, siblings):
    projected = select_headers(tree, example["columns"], schema, intent)
    if projected is None:
        return
    pool = [(projected, ("align_projection",), .70)]
    for edit in list(grounded_predicate(projected, schema, intent))[:2]:
        pool.append((edit.tree, ("align_projection", "bind_literal_role"), .93))
    for sibling in siblings[:4]:
        for changed in filter_transplants(projected, sibling, schema, intent, request.quoted):
            pool.append((changed, ("bind_hint_filter", "attach_filter_owner"), .84))
            if len(pool) >= 5:
                break
        if len(pool) >= 5:
            break
    for base, operations, strength in pool[:5]:
        emitted = False
        for rebuilt in reconstruct(base, schema, intent, example):
            if key(sql_text(rebuilt)) != key(sql_text(tree)):
                emitted = True
                yield rebuilt, operations + ("rebuild_relationships",), strength
        if not emitted and key(sql_text(base)) != key(sql_text(tree)):
            yield base, operations, strength
        # Existing relationship edits can preserve a legitimate non-FK portion of the query.
        for edit in list(relationships(base, schema))[:1]:
            yield edit.tree, operations + ("repair_local_relationship",), strength - .04


def statistic_target(query, schema, intent):
    phrase = re.search(r"\b(?:number of|count of|how many|most|fewest)\s+([^,.?]{2,65})", intent.text)
    if not phrase:
        return None
    words = parts(re.split(r"\b(?:with|who|that|which|from|in|for|by|per|are|is|have|has)\b", phrase[1], 1)[0])
    words -= {"the", "all", "total", "different", "distinct", "unique", "number"}
    hits = [(len(parts(display)), relation) for relation, display in schema.tables.items()
            if parts(display) and parts(display) <= words]
    if not hits:
        return None
    best = max(size for size, _ in hits)
    options = [relation for size, relation in hits if size == best]
    return options[0] if len(options) == 1 else None


def aggregate_structures(tree, schema, intent, request, example):
    tags = cues(intent.text)
    ranking = "rank" in tags and ("count" in tags or bool(re.search(r"\bmost\s+\w+", intent.text)))
    per_group = "group" in tags and ("count" in tags or tree.find(exp.Count))
    by_count = bool(re.search(r"\b(?:order|sort)\w*\b[^.?!]{0,40}\b(?:number|count)\b", intent.text))
    if not (ranking or per_group or by_count or request.boundary and tree.find(exp.Having)):
        return
    projected = select_headers(tree, example["columns"], schema, intent)
    if projected is None:
        return
    frame = build_frame(projected, schema, intent, example)
    if frame is None:
        return
    outputs = frame.select.expressions
    keys = [p.unalias().copy() for p in outputs if not p.find(exp.AggFunc)]
    if not keys:
        return
    target = statistic_target(projected, schema, intent)
    candidates = [frame.select]
    if target and target != frame.population:
        copy = frame.select.copy()
        if schema.attach(copy, target):
            candidates.insert(0, copy)
    elif ranking or per_group:
        present = {table.name.casefold() for table in sources(projected)}
        others = [(len(parts(display) & intent.words), relation) for relation, display in schema.tables.items()
                  if relation != frame.population and (relation in present or parts(display) & intent.words)
                  and schema.path([frame.population], relation)]
        if others:
            relation = max(others)[1]
            copy = frame.select.copy()
            if schema.attach(copy, relation):
                candidates.insert(0, copy)
    for base in candidates[:2]:
        count = exp.Count(this=exp.Star())
        if target and schema.primary(target) and target in {t.name.casefold() for t in sources(base)}:
            field = column(schema.primary(target), schema.tables[target])
            count = exp.Count(this=exp.Distinct(expressions=[field])) if intent.distinct else exp.Count(this=field)
        variants = [count]
        if target and not intent.distinct and not isinstance(count.this, exp.Star):
            variants.append(exp.Count(this=exp.Distinct(expressions=[count.this.copy()])))
        for aggregate in variants[:2]:
            changed = base.copy()
            projected_fields = []
            for node in changed.expressions:
                expression = node.copy()
                if isinstance(expression, exp.Count):
                    expression = aggregate.copy()
                else:
                    for original in list(expression.find_all(exp.Count)):
                        original.replace(aggregate.copy())
                projected_fields.append(expression)
            changed.set("expressions", projected_fields)
            changed.set("distinct", None if any(p.find(exp.AggFunc) for p in projected_fields) else changed.args.get("distinct"))
            changed.group_by(*[p.copy() for p in keys], append=False, copy=False)
            if request.boundary and tree.find(exp.Having):
                operator, value = request.boundary[0]
                changed.having(operator(this=aggregate.copy(), expression=exp.Literal.number(value)), append=False, copy=False)
            if ranking or by_count:
                ascending = bool(re.search(r"\b(?:least|fewest|minimum|smallest|lowest|bottom|ascending)\b", intent.text))
                changed.order_by(exp.Ordered(this=aggregate.copy(), desc=not ascending), append=False, copy=False)
                if ranking and not by_count:
                    changed.limit(rank_limit(intent.text), copy=False)
            yield changed, ("bind_statistical_population", "set_group_key", "count_order_and_limit"), .91


def constructions(agent):
    """Static work is bounded and uses at most five raw sources plus two checked anchors."""
    originals = [(agent.branches[0].sql, "original")]
    originals += [(sql, f"hint-{index}") for sql, index, _ in agent.hints.values()]
    originals += [(branch.sql, "checked") for branch in (agent.best(), agent.branches[-1])]
    seen = set()
    inputs = []
    for sql, label in originals:
        identity = key(sql)
        if identity not in seen:
            seen.add(identity)
            tree = parse(sql)
            if isinstance(tree, exp.Select) and not tree.args.get("with_"):
                inputs.append((sql, label, tree))
    siblings = [tree for _, _, tree in inputs]
    candidates = {}
    for sql, label, tree in inputs[:7]:
        paths = []
        if agent.config.get("joint_filter"):
            paths.append(("joint_filter", joined_filters(tree, agent.schema, agent.intent, agent.request, agent.example, siblings)))
        if agent.config.get("joint_stats"):
            paths.append(("joint_stats", aggregate_structures(tree, agent.schema, agent.intent, agent.request, agent.example)))
        for kind, proposals in paths:
            for proposal, operations, evidence in proposals:
                statement = sql_text(proposal)
                identity = key(statement)
                if identity in seen or identity in candidates or parse(statement) is None or invalid_bindings(proposal, agent.schema):
                    continue
                if isinstance(proposal, exp.Select) and len(proposal.expressions) != len(agent.example["columns"]):
                    continue
                dependencies = tuple(sorted({t.name.casefold() for s in proposal.find_all(exp.Select) for t in sources(s)}))
                if any(table not in agent.schema.tables for table in dependencies):
                    continue
                gain = header_agreement(proposal, agent.example["columns"]) if isinstance(proposal, exp.Select) else 0.
                candidates[identity] = Construction(proposal, kind, sql, operations, dependencies,
                                                    evidence + .12 * gain - .008 * len(dependencies))
                if len(candidates) >= agent.config.get("joint_internal_cap", 24):
                    break
            if len(candidates) >= agent.config.get("joint_internal_cap", 24):
                break
        if len(candidates) >= agent.config.get("joint_internal_cap", 24):
            break
    ordered = sorted(candidates.values(), key=lambda p: (-p.evidence, key(sql_text(p.tree))))
    # Interleave the two paths so a source cannot fill the entire bounded menu.
    chosen = []
    for offset in range(4):
        for kind in ("joint_filter", "joint_stats"):
            family = [proposal for proposal in ordered if proposal.family == kind]
            if len(family) > offset:
                chosen.append(family[offset])
    return chosen[:agent.config.get("joint_retained", 8)]
