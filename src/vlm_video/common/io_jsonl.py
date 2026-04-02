"""JSONL (newline-delimited JSON) read / write utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write *records* to a JSONL file, overwriting any existing content.

    Parameters
    ----------
    path:
        Destination file path.
    records:
        Iterable of JSON-serialisable dictionaries.

    Returns
    -------
    int
        Number of records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file one at a time.

    Parameters
    ----------
    path:
        Source file path.

    Yields
    ------
    dict
        Parsed JSON object per line.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single *record* to a JSONL file (creates the file if absent).

    Parameters
    ----------
    path:
        Destination file path.
    record:
        JSON-serialisable dictionary to append.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
