# src/signal_utils.py - Small numeric helpers shared by feature extractors
def longest_sustained_run(values, threshold: float) -> tuple[int, int]:
    """Longest run of consecutive values that stay on the same side of
    +/-threshold. Returns (run_length, sign): sign is +1 (positive side
    held it), -1 (negative side held it), or 0 (no run)."""
    best_len, best_sign = 0, 0
    cur_len, cur_sign = 0, 0
    for v in values:
        sign = 1 if v >= threshold else (-1 if v <= -threshold else 0)
        if sign != 0 and sign == cur_sign:
            cur_len += 1
        elif sign != 0:
            cur_sign, cur_len = sign, 1
        else:
            cur_sign, cur_len = 0, 0
        if cur_len > best_len:
            best_len, best_sign = cur_len, cur_sign
    return best_len, best_sign
