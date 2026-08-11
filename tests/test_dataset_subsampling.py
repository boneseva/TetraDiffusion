import pytest
from lib.Tetradata import select_prefix


def test_select_prefix_properties():
    paths = [f"path_{i:03d}" for i in range(100)]

    p25 = select_prefix(paths, 0.25, seed=42)
    assert len(p25) == 25

    p50 = select_prefix(paths, 0.50, seed=42)
    assert len(p50) == 50

    p75 = select_prefix(paths, 0.75, seed=42)
    assert len(p75) == 75

    p100 = select_prefix(paths, 1.0, seed=42)
    assert len(p100) == 100

    # 1. Nesting tests: 25% ⊂ 50% ⊂ 75% ⊂ 100%
    assert set(p25).issubset(set(p50))
    assert set(p50).issubset(set(p75))
    assert set(p75).issubset(set(p100))

    # 2. Length check for 100%
    assert len(p100) == len(paths)

    # 3. Determinism test
    assert select_prefix(paths, 0.25, seed=42) == p25


def test_invalid_fraction_and_empty():
    paths = [f"path_{i:03d}" for i in range(10)]

    # Out of range fraction
    with pytest.raises(ValueError, match=r"dataset_fraction must be in \(0, 1\.0\]"):
        select_prefix(paths, 0.0, seed=42)

    with pytest.raises(ValueError, match=r"dataset_fraction must be in \(0, 1\.0\]"):
        select_prefix(paths, 1.5, seed=42)

    # Empty paths list
    with pytest.raises(ValueError, match=r"paths list cannot be empty"):
        select_prefix([], 0.25, seed=42)
