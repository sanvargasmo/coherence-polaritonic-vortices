import numpy as np

from polaritonic_vortices import ProductBasis, triangular_count, triangular_index


def test_triangular_count():
    assert triangular_count(1) == 3
    assert triangular_count(8) == 45
    assert triangular_count(10) == 66


def test_product_basis_dimensions():
    assert ProductBasis(1, 1).dimension == 9
    assert ProductBasis(8, 8).dimension == 2025
    assert ProductBasis(10, 10).dimension == 4356


def test_triangular_index_is_unique():
    basis = ProductBasis(8, 8)
    indices = [basis.cavity_index(n, m) for n, m in basis.cavity_pairs()]
    assert sorted(indices) == list(range(45))


def test_product_basis_index_matches_original_nested_loop_order():
    cutoff = 4
    basis = ProductBasis(cutoff, cutoff)
    original_order = [
        (na, ma, nb, mb)
        for na in range(cutoff + 1)
        for ma in range(cutoff + 1) if na + ma <= cutoff
        for nb in range(cutoff + 1)
        for mb in range(cutoff + 1) if nb + mb <= cutoff
    ]
    reconstructed = [None] * basis.dimension
    for state in original_order:
        reconstructed[basis.state_index(*state)] = state
    assert reconstructed == original_order
