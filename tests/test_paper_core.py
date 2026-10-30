import numpy as np
from polaritonic_vortices import ExactCoherentEvolution, SimulationConfig
from polaritonic_vortices.paper_core import GaussianRemovedProfile, track_paper_core


def test_fixed_gaussian_factor_reconstructs_full_density():
    e=ExactCoherentEvolution(SimulationConfig(mean_photon_number=1.1))
    w=e.physical.oscillator_width
    for t in [0.,.23,.71]:
        f=GaussianRemovedProfile(e,t,'cavity')
        for r in [np.array([.2,-.3]),np.array([1.,.7])]:
            density=e.spatial_density_grid(t,'cavity',[r[0]*w],[r[1]*w])[0,0]*w*w
            np.testing.assert_allclose(f.value(r)*np.exp(-np.dot(r,r)),density,atol=1e-12)


def test_initial_core_for_zero_and_large_coherent_backgrounds():
    for a in [0.,.005,.5,1.1]:
        e=ExactCoherentEvolution(SimulationConfig(mean_photon_number=a))
        for sub in ['cavity','exciton']:
            f=GaussianRemovedProfile(e,0.,sub)
            p=e.physical
            core=np.array([getattr(p,sub+'_x'),getattr(p,sub+'_y')])/p.oscillator_width
            candidates=f.minima([core-f.mu])
            assert len(candidates)==1
            np.testing.assert_allclose(candidates[0][0],core,atol=1e-10)
            assert abs(f.value(core))<1e-10


def test_fixed_factor_stationary_points_and_true_local_minima():
    e=ExactCoherentEvolution(SimulationConfig(mean_photon_number=.005))
    times=np.linspace(0,.3,9)
    tracks=track_paper_core(e,times,'cavity')
    for t,r in zip(times,tracks['xy_over_w']):
        if not np.isfinite(r).all():continue
        f=GaussianRemovedProfile(e,t,'cavity')
        h=1e-4
        gradient=np.array([(f.value(r+h*v)-f.value(r-h*v))/(2*h) for v in np.eye(2)])
        np.testing.assert_allclose(gradient,0,atol=1e-7)
        for v in [np.array([1.,0.]),np.array([0.,1.]),np.array([1.,1.])/np.sqrt(2)]:
            assert f.value(r+h*v)+f.value(r-h*v)-2*f.value(r)>0


def test_envelope_choices_coincide_at_alpha_zero():
    e=ExactCoherentEvolution(SimulationConfig(mean_photon_number=0))
    for t in [0.,.2,.7]:
        f=GaussianRemovedProfile(e,t,'cavity','fixed')
        c=GaussianRemovedProfile(e,t,'cavity','coherent')
        np.testing.assert_allclose(f.g,c.g,atol=1e-14)
