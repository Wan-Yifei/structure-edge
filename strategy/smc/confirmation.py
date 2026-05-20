"""LTF entry confirmation: CHoCH + BOS sequence detection."""

from __future__ import annotations


def check_ltf_confirmation(
    ltf_bos: list[dict],
    trend: str,
    after_idx: int,
) -> bool:
    """Return True if LTF signals show CHoCH + BOS in *trend* direction after *after_idx*.

    Sequence required: at least one CHoCH in the trend direction, followed by
    at least one BOS in the same direction — both occurring after after_idx.
    """
    relevant = [s for s in ltf_bos if s["idx"] > after_idx]
    if not relevant:
        return False

    choch_idx = -1
    for sig in relevant:
        if sig["type"] == "CHoCH" and sig["direction"] == trend:
            choch_idx = sig["idx"]
            break

    if choch_idx < 0:
        return False

    for sig in relevant:
        if sig["idx"] > choch_idx and sig["type"] == "BOS" and sig["direction"] == trend:
            return True

    return False
