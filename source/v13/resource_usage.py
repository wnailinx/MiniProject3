"""Linux process-tree RSS sampling, independent of historical workspace files."""
import os
from pathlib import Path


def memory_tree(root_pid):
    records = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")",1)[1].split()
            records[int(path.name)] = (int(fields[1]),int(fields[20]),int(fields[21]))
        except (FileNotFoundError,PermissionError,ProcessLookupError,ValueError):
            continue
    family = {root_pid}
    while True:
        expanded = family | {pid for pid,value in records.items() if value[0] in family}
        if expanded == family:
            break
        family = expanded
    rss = sum(records[pid][2]*os.sysconf("SC_PAGE_SIZE") for pid in family if pid in records)
    return rss,records.get(root_pid,(0,0,0))[1]
