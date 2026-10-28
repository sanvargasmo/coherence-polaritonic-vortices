"""Cutoff-free evolution for the repository's vortex-on-coherent-state family.

Write Psi(0)=D(alpha_C)D(alpha_X)|s(0)>. Multiplying each coherent
Gaussian by a linear vortex factor leaves at most one excitation in each
DISPLACED subsystem. The number-conserving quadratic Hamiltonian propagates
the displacement with its single-particle matrix, while |s(t)> stays in the
TOTAL-excitation <=2 subspace. A local cutoff of two therefore represents
the displaced seed exactly at every time, for arbitrary coherent amplitude.
The physical state still has infinitely many Fock components; it is NOT
approximated by a two-excitation state in the undisplaced basis.

Coordinates refer to the common rotating frame specified in dynamics.py.
"""
from math import factorial
import numpy as np
from scipy.linalg import eigh
from .basis import ProductBasis
from .parameters import PhysicalParameters, SimulationConfig
from .dynamics import rotating_frame_hamiltonian
from .reduced_density import linear_entropy
from .spatial_fields import _basis_grid


class ExactCoherentEvolution:
    def __init__(self, config=None, physical=None):
        self.config = config or SimulationConfig()
        self.physical = physical or PhysicalParameters()
        self.basis = ProductBasis(2, 2)
        self.pairs = list(self.basis.cavity_pairs())
        self.alpha = 0j if self.config.use_bare_vortex_initial_state else self.config.alpha
        p = self.physical
        mu = np.sqrt(2) * p.oscillator_width * self.alpha.real
        local = []
        for x0, y0 in [(p.cavity_x, p.cavity_y), (p.exciton_x, p.exciton_y)]:
            beta = (mu-x0) + 1j*(mu-y0)
            den = np.sqrt(p.oscillator_width**2 + abs(beta)**2)
            c = np.zeros(6, complex)
            c[self.pairs.index((0, 0))] = beta / den
            c[self.pairs.index((1, 0))] = p.oscillator_width / np.sqrt(2) / den
            c[self.pairs.index((0, 1))] = 1j*p.oscillator_width / np.sqrt(2) / den
            local.append(c)
        self.initial_seed = np.outer(*local).ravel()
        self.energies, self.eigenvectors = eigh(rotating_frame_hamiltonian(self.basis, p).toarray())
        self.weights = self.eigenvectors.conj().T @ self.initial_seed
        self.delta = p.delta_omega_ps_inv
        self.g = np.hypot(p.omega_ps_inv, self.delta/2)

    def seed_matrix(self, time):
        c = self.eigenvectors @ (np.exp(-1j*self.energies*float(time))*self.weights)
        return c.reshape(6, 6)

    def displacement(self, time):
        """Return alpha_Cx,Cy,Xx,Xy in the common rotating frame."""
        p = self.physical
        h = np.array([[self.delta/2, p.omega_ps_inv],
                      [p.omega_ps_inv, -self.delta/2]])
        u = np.cos(self.g*time)*np.eye(2)-1j*np.sin(self.g*time)*h/self.g
        ac, ax = u @ np.array([self.alpha, self.alpha])
        return np.array([ac, ac, ax, ax])

    def reduced_seed(self, time, subsystem):
        c = self.seed_matrix(time)
        if subsystem == 'cavity':
            return c @ c.conj().T
        if subsystem == 'exciton':
            return c.T @ c.conj()
        raise ValueError('subsystem must be cavity or exciton')

    def entropy(self, time):
        # Local displacement operators preserve the Schmidt spectrum.
        return linear_entropy(self.reduced_seed(time, 'cavity'))

    def angular_momentum(self, time, subsystem):
        rho = self.reduced_seed(time, subsystem)
        index = {pair: i for i, pair in enumerate(self.pairs)}
        annihilators = []
        offset = 0 if subsystem == 'cavity' else 2
        for axis, alpha in enumerate(self.displacement(time)[offset:offset+2]):
            a = np.zeros((6, 6), complex)
            for j, pair in enumerate(self.pairs):
                if pair[axis]:
                    lower = list(pair); lower[axis] -= 1
                    a[index[tuple(lower)], j] = np.sqrt(pair[axis])
            annihilators.append(a + alpha*np.eye(6))
        ax, ay = annihilators
        lz = 1j*(ay.conj().T @ ax - ax.conj().T @ ay)
        return float(np.trace(rho @ lz).real)

    def spatial_density_grid(self, time, subsystem, xs, ys):
        offset = 0 if subsystem == 'cavity' else 2
        mu = np.sqrt(2)*self.physical.oscillator_width*self.displacement(time)[offset:offset+2].real
        phi = _basis_grid(self.pairs, np.asarray(xs)-mu[0], np.asarray(ys)-mu[1], self.physical.oscillator_width)
        rho = self.reduced_seed(time, subsystem)
        return np.einsum('...i,ij,...j->...', phi, rho, phi, optimize=True).real

    def fock_matrix(self, time, cutoff):
        """Exact-state projection onto an undisplaced local triangular basis.

        No renormalization: its norm measures missing probability and allows
        independent comparison against the numerical Fock-space solver.
        """
        pairs = list(ProductBasis(cutoff, cutoff).cavity_pairs())
        aa = self.displacement(time)
        transforms = []
        for offset in [0, 2]:
            vectors = []
            for alpha in aa[offset:offset+2]:
                c = np.array([np.exp(-abs(alpha)**2/2)*alpha**n/np.sqrt(float(factorial(n))) for n in range(cutoff+1)])
                v1 = np.zeros_like(c); v1[1:] = np.sqrt(np.arange(1, cutoff+1))*c[:-1]
                v2 = np.zeros_like(c); v2[2:] = np.sqrt(np.arange(2, cutoff+1)*np.arange(1, cutoff))*c[:-2]
                vectors.append([c, v1-alpha.conjugate()*c,
                                (v2-2*alpha.conjugate()*v1+alpha.conjugate()**2*c)/np.sqrt(2)])
            transforms.append(np.array([[vectors[0][n][i]*vectors[1][m][j] for n,m in self.pairs] for i,j in pairs]))
        return transforms[0] @ self.seed_matrix(time) @ transforms[1].T
