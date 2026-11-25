"""SQL shape, public output contract, and schema bindings."""
from collections import Counter, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import re

import sqlglot
from sqlglot import exp


@lru_cache(maxsize=2048)
def parse(sql):
    if not sql or len(sql.encode()) > 16000:
        return None
    try:
        statements = sqlglot.parse(sql, read="sqlite")
        if len(statements) == 1 and isinstance(statements[0], exp.Query):
            return statements[0]
    except (sqlglot.errors.SqlglotError, RecursionError, ValueError):
        pass
    return None


def sql_text(tree):
    return tree.sql(dialect="sqlite", comments=False)


@lru_cache(maxsize=4096)
def key(sql):
    tree = parse(sql)
    if tree is None:
        return sql.strip()
    tree = tree.copy()
    if isinstance(tree, exp.Select) and len(list(tree.find_all(exp.Select))) == 1:
        tables = sources(tree)
        if tables and len({t.name.casefold() for t in tables}) == len(tables):
            aliases = {t.alias_or_name.casefold(): t.this for t in tables}
            for item in local(tree, exp.Column):
                if item.table.casefold() in aliases:
                    item.set("table", aliases[item.table.casefold()].copy())
            for table in tables:
                table.set("alias", None)
    return tree.sql(dialect="sqlite", normalize=True, comments=False)


def parts(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    tokens = re.findall(r"[a-z]+|\d+", text.lower())
    return {word[:-1] if len(word) > 3 and word.endswith("s") else word for word in tokens}


def norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def local(select, kind):
    return [node for node in select.find_all(kind) if node.find_ancestor(exp.Select) is select]


def sources(select):
    tables = []
    source = select.args.get("from")
    if source is None:
        source = select.args.get("from_")
    if source is not None and isinstance(source.this, exp.Table):
        tables.append(source.this)
    tables.extend(join.this for join in select.args.get("joins", ()) if isinstance(join.this, exp.Table))
    return tables


def label(expression):
    expression = expression.unalias()
    if isinstance(expression, exp.Column):
        return norm(expression.name)
    normalized = expression.transform(
        lambda node: exp.column(node.name) if isinstance(node, exp.Column) else node)
    return norm(sql_text(normalized))


@lru_cache(maxsize=512)
def header_label(header):
    tree = parse("SELECT " + header)
    if isinstance(tree, exp.Select) and len(tree.expressions) == 1:
        return label(tree.expressions[0])
    return norm(header)


def header_agreement(select, headers):
    actual = [label(column) for column in select.expressions]
    expected = [header_label(header) for header in headers]
    if not expected:
        return 0.0
    aligned = sum(a == b for a, b in zip(actual, expected))
    overlap = sum((Counter(actual) & Counter(expected)).values())
    return (2 * aligned + overlap) / (3 * len(expected)) - .35 * abs(len(actual) - len(expected))


@dataclass(frozen=True)
class Intent:
    text: str
    words: frozenset
    count: bool
    distinct: bool
    negative: bool
    grouped: bool
    frequent: bool
    descending: bool | None
    numbers: tuple

    @classmethod
    def read(cls, text):
        low = text.lower()
        clean = re.sub(r"\b(at least|at most|no more than|no fewer than)\b", "", low)
        count = bool(re.search(r"\b(how many|number of|count of)\b", low))
        frequent = bool(re.search(r"\b(most common|least common|most frequent|least frequent|"
                                  r"most popular|largest number|greatest number|smallest number|most number|least number|"
                                  r"occur\w* most|most often|fewest)\b", low))
        descending = None
        if re.search(r"\b(lowest|fewest|smallest|earliest|ascending|least common|least frequent|alphabetical)\b", clean):
            descending = False
        elif re.search(r"\b(highest|most|largest|latest|descending|greatest|top)\b", clean):
            descending = True
        return cls(low, frozenset(parts(text)), count,
                   bool(re.search(r"\b(distinct|unique|different)\b", low)),
                   bool(re.search(r"\b(not|never|without|except|no)\b", low)),
                   bool(re.search(r"\b(each|per|by each|for every)\b", low)),
                   frequent, descending, tuple(re.findall(r"\b\d+(?:\.\d+)?\b", low)))


class Schema:
    def __init__(self, records):
        self.tables = {r["name"].casefold(): r["name"] for r in records}
        self.columns = {r["name"].casefold(): {c["name"].casefold(): c for c in r["columns"]}
                        for r in records}
        self.edges = {name: [] for name in self.tables}
        for r in records:
            source = r["name"].casefold()
            for fk in r["foreign_keys"]:
                target = fk["table"].casefold()
                right = fk["to"] or self.primary(target)
                if target in self.tables and right:
                    self.edges[source].append((target, fk["from"], right))
                    self.edges[target].append((source, right, fk["from"]))

    def primary(self, table):
        return next((c["name"] for c in self.columns.get(table, {}).values() if c["pk"] == 1), None)

    def resolve_table(self, name):
        direct = self.tables.get(name.casefold())
        if direct:
            return direct
        match = [t for t in self.tables.values() if parts(t) == parts(name)]
        return match[0] if len(match) == 1 else None

    def bindings(self, select, name, intent, qualifier="", *, fuzzy=False):
        """Rank current-schema bindings using names, scope and question terms."""
        scope = {table.name.casefold(): table.alias_or_name for table in sources(select)}
        result = []
        for table, columns in self.columns.items():
            for column in columns.values():
                actual = column["name"]
                exact = norm(actual) == norm(name)
                similarity = 1.0 if exact else SequenceMatcher(None, norm(name), norm(actual)).ratio()
                if not exact and (not fuzzy or similarity < .65):
                    continue
                alias = scope.get(table, self.tables[table])
                score = 3 * similarity + 1.3 * (table in scope)
                score += .30 * len(parts(self.tables[table]) & intent.words)
                if qualifier:
                    score += 2.0 * (qualifier.casefold() in {alias.casefold(), table})
                result.append((score, table, alias, actual))
        return sorted(result, key=lambda item: (-item[0], item[1], item[3]))

    def path(self, start, target):
        if target in start:
            return []
        queue = deque((name, []) for name in start)
        seen = set(start)
        while queue:
            source, route = queue.popleft()
            if len(route) >= 3:
                continue
            for neighbor, left, right in self.edges.get(source, ()):
                if neighbor in seen:
                    continue
                path = route + [(source, neighbor, left, right)]
                if neighbor == target:
                    return path
                queue.append((neighbor, path))
                seen.add(neighbor)
        return None

    def attach(self, select, target):
        scope = {table.name.casefold(): table.alias_or_name for table in sources(select)}
        path = self.path(scope, target)
        if path is None:
            return False
        for old, new, left, right in path:
            table = self.tables[new]
            select.join(exp.Table(this=identifier(table)),
                        on=exp.EQ(this=column(left, scope[old]), expression=column(right, table)),
                        copy=False)
            scope[new] = table
        return True


def identifier(name):
    return exp.to_identifier(name, quoted=not bool(re.fullmatch(r"[A-Za-z_][\w]*", name)))


def column(name, table=None):
    return exp.Column(this=identifier(name), table=identifier(table) if table else None)


def invalid_bindings(tree, schema):
    """Conservative static checks in simple scopes with known base tables."""
    if not isinstance(tree, exp.Select) or tree.args.get("with") or tree.args.get("with_"):
        return 0
    count = 0
    for select in tree.find_all(exp.Select):
        scope = sources(select)
        if not scope or len(scope) != len(select.args.get("joins", [])) + 1:
            continue
        if any(t.name.casefold() not in schema.tables for t in scope):
            count += sum(t.name.casefold() not in schema.tables for t in scope)
            continue
        aliases = {t.alias_or_name.casefold(): schema.columns[t.name.casefold()] for t in scope}
        output_aliases = {expression.alias.casefold() for expression in select.expressions if expression.alias}
        nested = select is not tree
        for node in local(select, exp.Column):
            if isinstance(node.this, exp.Star) or not node.name or node.name.casefold() in output_aliases:
                continue
            alias = node.table.casefold()
            name = node.name.casefold()
            if alias:
                count += (alias in aliases and name not in aliases[alias]) or (alias not in aliases and not nested)
            elif not nested and not node.this.args.get("quoted") and name not in {"rowid", "_rowid_", "oid"}:
                count += not any(name in columns for columns in aliases.values())
    return min(count, 5)


@dataclass(frozen=True)
class Diagnosis:
    error: str
    matched: bool
    empty: bool
    rows: str
    projection: str
    aggregate: bool
    join: bool
    predicate: bool
    grouped: bool
    negative: bool

    @classmethod
    def inspect(cls, tree, feedback, example):
        message = (feedback.get("error") or {}).get("message", "").lower()
        kind = ("column" if "column" in message else "table" if "table" in message else
                "aggregate" if "aggregate" in message else "syntax" if message else "none")
        expected = example["row_count"]
        actual = feedback.get("row_count", 0)
        headers = example.get("columns", [])
        score = header_agreement(tree, headers) if isinstance(tree, exp.Select) else 0.0
        arity = len(feedback.get("columns", [])) if feedback.get("ok") else None
        delta = actual - expected
        observed_rows, expected_rows = feedback.get("rows", []), example.get("rows", [])
        if (delta == 0 and actual == 1 and observed_rows and expected_rows
                and len(observed_rows[0]) == len(expected_rows[0]) == 1
                and all(isinstance(value, (float, int)) for value in (observed_rows[0][0], expected_rows[0][0]))
                and any("count" in header.lower() for header in headers)):
            delta = observed_rows[0][0] - expected_rows[0][0]
        return cls(kind, bool(feedback.get("matches_example")), not expected,
                   "more" if delta > 0 else "fewer" if delta < 0 else "same",
                   "arity" if arity is not None and arity != len(headers) else
                   "different" if score < .98 else "aligned",
                   bool(tree and tree.find(exp.AggFunc)),
                   bool(tree and tree.find(exp.Join)), bool(tree and tree.find(exp.Where)),
                   bool(tree and tree.find(exp.Group)), bool(tree and (tree.find(exp.Not) or tree.find(exp.Except))))
