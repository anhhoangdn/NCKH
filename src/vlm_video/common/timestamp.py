"""Timestamp conversion utilities (seconds ↔ HH:MM:SS)."""

from __future__ import annotations

import re


def sec_to_hms(seconds: float) -> str:
    """Convert *seconds* to an ``HH:MM:SS.mmm`` string.

    Parameters
    ----------
    seconds:
        Non-negative number of seconds.

    Returns
    -------
    str
        Formatted timestamp, e.g. ``"01:23:45.678"``.

    Examples
    --------
    >>> sec_to_hms(3725.5)
    '01:02:05.500'
    """
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")
    millis = round(seconds * 1000)
    ms = millis % 1000
    total_secs = millis // 1000
    h = total_secs // 3600
    m = (total_secs % 3600) // 60
    s = total_secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def hms_to_sec(hms_str: str) -> float:
    """Convert an ``HH:MM:SS[.mmm]`` string to seconds.

    Parameters
    ----------
    hms_str:
        Timestamp string in ``HH:MM:SS``, ``HH:MM:SS.mmm``, ``MM:SS``, or
        ``MM:SS.mmm`` format.

    Returns
    -------
    float
        Total seconds.

    Examples
    --------
    >>> hms_to_sec("01:02:05.500")
    3725.5
    """
    pattern = r"^(?:(\d+):)?(\d+):(\d+)(?:\.(\d+))?$"
    match = re.match(pattern, hms_str.strip())
    if not match:
        raise ValueError(f"Cannot parse timestamp string: {hms_str!r}")

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    secs = int(match.group(3))
    frac_str = match.group(4) or "0"
    frac = int(frac_str) / (10 ** len(frac_str))

    return hours * 3600 + minutes * 60 + secs + frac


def format_timestamp(seconds: float) -> str:
    """Return a human-readable timestamp label suitable for display.

    For durations < 1 hour, returns ``MM:SS``; otherwise ``HH:MM:SS``.

    Parameters
    ----------
    seconds:
        Non-negative number of seconds.

    Returns
    -------
    str
        Compact timestamp string.

    Examples
    --------
    >>> format_timestamp(125.0)
    '02:05'
    >>> format_timestamp(3725.0)
    '01:02:05'
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
