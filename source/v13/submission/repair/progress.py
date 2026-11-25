"""Compare complete public previews to choose a repair branch."""
from collections import Counter
import math


def cell(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number", round(value, 6)
    return type(value).__name__, value


def fraction_agree(feedback, example):
    """Zero means the preview does not supply useful comparable evidence."""
    if (not feedback.get("ok") or feedback.get("truncated") or example.get("truncated")
            or not feedback.get("rows") or not example.get("rows") or not example.get("columns")):
        return 0.0
    actual, wanted = feedback["rows"], example["rows"]
    width = len(example["columns"])
    if any(len(row) != width for row in actual + wanted):
        return 0.0
    if width == 1 and len(actual) == len(wanted) == 1:
        x, y = actual[0][0], wanted[0][0]
        if (isinstance(x, (int, float)) and isinstance(y, (int, float))
                and math.isfinite(x) and math.isfinite(y)):
            return 1. / (1. + abs(x - y) / max(1., abs(y)))
    total = max(len(actual), len(wanted))
    ah = Counter(tuple(map(cell, row)) for row in actual)
    wh = Counter(tuple(map(cell, row)) for row in wanted)
    exact = sum((ah & wh).values()) / total
    aligned = 0.
    for index in range(width):
        actual_col = Counter(cell(row[index]) for row in actual)
        wanted_col = Counter(cell(row[index]) for row in wanted)
        aligned += sum((actual_col & wanted_col).values()) / total
    return .6 * exact + .4 * aligned / width
