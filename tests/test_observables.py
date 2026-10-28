import numpy as np

from polaritonic_vortices import (
    cavity_density_matrix,
    exciton_density_matrix,
    linear_entropy,
)
from polaritonic_vortices.observables import cavity_lz, exciton_lz, total_lz


def test_linear_entropy_matches_original_trrho_schro_formula(n05_short_result):
    # Original notebooks define trrhoSchro = sum rho_ij rho_ji = Tr(rho^2).
    for t in (0.0, 0.3, 1.0):
        rho_c = cavity_density_matrix(n05_short_result, t)
        direct = 1.0 - np.real(np.sum(rho_c * rho_c.T))
        assert np.isclose(linear_entropy(rho_c), direct, atol=2e-13)


def test_lz_matches_operator_expectation(n05_short_result):
    r=n05_short_result
    pairs=list(r.basis.cavity_pairs()); index={p:i for i,p in enumerate(pairs)}
    ax=np.zeros((len(pairs),len(pairs)),complex); ay=ax.copy()
    for j,(n,m) in enumerate(pairs):
        if n: ax[index[n-1,m],j]=np.sqrt(n)
        if m: ay[index[n,m-1],j]=np.sqrt(m)
    lz=1j*(ay.conj().T@ax-ax.conj().T@ay)
    for t in [0,.3,1.]:
        assert abs(cavity_lz(r,t)-np.trace(cavity_density_matrix(r,t)@lz))<1e-13
        assert abs(exciton_lz(r,t)-np.trace(exciton_density_matrix(r,t)@lz))<1e-13


def test_total_lz_is_conserved_in_n05_short_run(n05_short_result):
    values = np.array(
        [total_lz(n05_short_result, t).real for t in np.linspace(0.0, 1.0, 11)]
    )
    assert np.max(np.abs(values - values[0])) < 3e-9


def test_cavity_and_exciton_reduced_states_have_same_linear_entropy(n05_short_result):
    # For the same unnormalized global pure state, both reductions have the same purity.
    for t in (0.0, 0.37, 1.0):
        rho_c = cavity_density_matrix(n05_short_result, t)
        rho_x = exciton_density_matrix(n05_short_result, t)
        assert np.isclose(linear_entropy(rho_c), linear_entropy(rho_x), atol=2e-12)
