"""Independent physical checks; legacy notebook numbers are not ground truth."""
import numpy as np
import pytest
from scipy.linalg import expm
from numpy.polynomial.hermite import hermgauss
from polaritonic_vortices import *
from polaritonic_vortices.dynamics import rotating_frame_hamiltonian
from polaritonic_vortices.initial_state import InitialStateBuilder
from polaritonic_vortices.density_minima import DensityProfile


def test_numerical_state_matches_static_hamiltonian_exponential():
    r=simulate(SimulationConfig(cavity_cutoff=2,exciton_cutoff=2,mean_photon_number=0,t_max=.7,n_times=5))
    h=rotating_frame_hamiltonian(r.basis,r.physical).toarray()
    assert np.linalg.norm(r.state_at(.7)-expm(-1j*h*.7)@r.initial_state)<2e-8
    exact=ExactCoherentEvolution(r.config,r.physical)
    assert np.linalg.norm(exact.fock_matrix(.7,2).ravel()-r.state_at(.7))<2e-8


@pytest.mark.parametrize('nbar',[0,.005,.5,1.1])
def test_exact_initial_projection_matches_independent_initial_builder(nbar):
    cfg=SimulationConfig(mean_photon_number=nbar,cavity_cutoff=12,exciton_cutoff=12,renormalize_initial_state=False)
    b=ProductBasis(12,12)
    original=InitialStateBuilder(b,PhysicalParameters(),cfg).build()
    exact=ExactCoherentEvolution(cfg)
    assert np.linalg.norm(original-exact.fock_matrix(0,12).ravel())<2e-14
    assert abs(exact.entropy(0))<3e-14


def test_displaced_exact_evolution_matches_independent_fock_integration():
    # For this weak displacement cutoff 8 has negligible dynamic tail.
    cfg=SimulationConfig(mean_photon_number=.005,cavity_cutoff=8,exciton_cutoff=8,t_max=.7,n_times=5,rtol=1e-11,atol=1e-13)
    numerical=simulate(cfg)
    exact=ExactCoherentEvolution(cfg)
    assert np.linalg.norm(exact.fock_matrix(.7,8).ravel()-numerical.state_at(.7))<2e-6


@pytest.mark.parametrize('nbar',[0,.005,.5,1.1])
def test_normalization_entanglement_bounds_and_angular_momentum(nbar):
    e=ExactCoherentEvolution(SimulationConfig(mean_photon_number=nbar))
    total=[]
    for t in np.linspace(0,20,51):
        c=e.seed_matrix(t)
        assert abs(np.vdot(c,c).real-1)<1e-13
        rc=e.reduced_seed(t,'cavity'); rx=e.reduced_seed(t,'exciton')
        assert np.linalg.eigvalsh(rc).min()>-1e-13
        assert abs(linear_entropy(rc)-linear_entropy(rx))<1e-13
        assert -1e-13<=e.entropy(t)<=1
        total.append(e.angular_momentum(t,'cavity')+e.angular_momentum(t,'exciton'))
    assert np.ptp(total)<2e-13


def test_exact_spatial_density_matches_fock_projection_and_integrates_to_one():
    e=ExactCoherentEvolution(SimulationConfig(mean_photon_number=.5))
    t=.53; w=e.physical.oscillator_width
    # High-order quadrature on the displaced Gaussian, independent of grid extent.
    nodes,weights=hermgauss(8)
    for sub,off in [('cavity',0),('exciton',2)]:
        mu=np.sqrt(2)*w*e.displacement(t)[off:off+2].real
        grid=e.spatial_density_grid(t,sub,w*nodes+mu[0],w*nodes+mu[1])
        integral=w*w*np.sum(weights[:,None]*weights[None,:]*grid*np.exp(nodes[:,None]**2+nodes[None,:]**2))
        assert abs(integral-1)<2e-13
    c=e.fock_matrix(t,16); rho=c@c.conj().T
    from polaritonic_vortices.spatial_fields import _basis_grid
    phi=_basis_grid(list(ProductBasis(16,16).cavity_pairs()),np.array([-10.,20.]),np.array([5.,15.]),w)
    projected=np.einsum('...i,ij,...j->...',phi,rho,phi).real
    assert np.max(abs(projected-e.spatial_density_grid(t,'cavity',[-10.,20.],[5.,15.])))<2e-8


@pytest.mark.parametrize('nbar',[0,.005,.5,1.1])
def test_true_initial_minimum_and_full_density_hessian(nbar):
    e=ExactCoherentEvolution(SimulationConfig(mean_photon_number=nbar))
    for sub in ['cavity','exciton']:
        p=e.physical; expected=np.array([getattr(p,sub+'_x'),getattr(p,sub+'_y')])/p.oscillator_width
        track=track_density_minimum(e,[0.],sub)
        assert np.linalg.norm(track[0]-expected)<1e-8
        profile=DensityProfile(e,0,sub); z=track[0]-profile.mu
        assert abs(profile.density(z))<1e-13
        assert np.linalg.eigvalsh(profile.jacobian(z)).min()>0


def test_reported_minima_are_minima_of_the_full_density():
    e=ExactCoherentEvolution(SimulationConfig(mean_photon_number=.005))
    times=np.linspace(0,3,61)
    track=track_density_minimum(e,times,'cavity')
    for t,q in zip(times,track):
        if np.isfinite(q).all():
            profile=DensityProfile(e,t,'cavity'); z=q-profile.mu
            assert np.linalg.norm(profile.gradient_polynomial(z))<1e-8
            for direction in [np.array([1e-3,0]),np.array([0,1e-3])]:
                assert profile.density(z+direction)>=profile.density(z)-1e-12
                assert profile.density(z-direction)>=profile.density(z)-1e-12


def test_truncated_product_state_has_zero_normalized_entropy():
    r=simulate(SimulationConfig(mean_photon_number=1.1,cavity_cutoff=8,exciton_cutoff=8,t_max=.001,n_times=2))
    assert abs(r.initial_norm-1)<1e-14
    assert r.initial_retained_probability<.992
    assert abs(linear_entropy(cavity_density_matrix(r,0)))<1e-14


def test_cutoff_one_warns_even_when_initial_state_has_unit_norm():
    with pytest.warns(RuntimeWarning,match='can leave the local cutoff'):
        r=simulate(SimulationConfig(mean_photon_number=0,cavity_cutoff=1,exciton_cutoff=1,t_max=.001,n_times=2))
    assert abs(r.initial_retained_probability-1)<1e-14
