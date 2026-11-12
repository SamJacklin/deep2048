import numpy as np

from scripts.split_downgrade import split_weights


def index(*exponents: int) -> int:
    return sum(e << (4 * i) for i, e in enumerate(exponents))


def test_untrained_16384_patterns_copy_the_pattern_the_old_network_read() -> None:
    weights = np.zeros(16**3, dtype=np.float32)
    weights[index(13, 12, 2)] = 5.0  # what the old network read for 16384-8192-4 (its gap was 4096)
    weights[index(13, 0, 0)] = 7.0  # and for a lone 16384, whose gap is 8192
    weights[index(14, 13, 1)] = 3.0  # a 16384 pattern the old network did train
    weights[index(12, 11, 3)] = 9.0  # below 16384, so untouched

    filled = split_weights(weights, [3], old_exponent=14, new_exponent=15)

    assert weights[index(14, 13, 2)] == 5.0
    assert weights[index(14, 0, 0)] == 7.0
    assert weights[index(0, 14, 0)] == 0.0  # copies a pattern the old network never trained
    assert weights[index(14, 13, 1)] == 3.0
    assert weights[index(12, 11, 3)] == 9.0
    assert weights[index(15, 14, 2)] == 0.0  # 32768 patterns stay as they were
    # Every pattern topped by a 16384 except the trained one was considered
    assert filled == sum(1 for i in range(16**3) if max((i >> 4 * k) & 15 for k in range(3)) == 14) - 1
