"""Track verified local minima of the FULL reduced spatial density.

These are density-depression tracks, not an assertion of a phase singularity
for a mixed reduced state. Missing minima are represented by NaN gaps.
"""
from math import factorial
import numpy as np
from scipy.signal import convolve2d
from scipy.optimize import root
from numpy.polynomial.polynomial import polyval2d, polyder


def _shift(p, axis):
    out = np.zeros((p.shape[0]+(axis == 0), p.shape[1]+(axis == 1)))
    if axis == 0:
        out[1:, :] = p
    else:
        out[:, 1:] = p
    return out


def _add(a, b):
    out = np.zeros(tuple(max(i,j) for i,j in zip(a.shape,b.shape)))
    out[:a.shape[0], :a.shape[1]] += a
    out[:b.shape[0], :b.shape[1]] += b
    return out


class DensityProfile:
    """D*w^2=exp(-|z|^2)*P(z), z=(r-mu)/w, with exact quartic P."""
    def __init__(self, evolution, time, subsystem):
        rho = evolution.reduced_seed(time, subsystem)
        polys = [np.array([1.]),np.array([0.,2.]),np.array([-2.,0.,4.])]
        basis = [np.outer(polys[n],polys[m])/np.sqrt(np.pi*2**(n+m)*factorial(n)*factorial(m))
                 for n,m in evolution.pairs]
        self.p = np.zeros((5,5))
        for i,a in enumerate(basis):
            for j,b in enumerate(basis):
                c = rho[i,j].real*convolve2d(a,b)
                self.p[:c.shape[0],:c.shape[1]] += c
        self.gx = _add(polyder(self.p,axis=0),-2*_shift(self.p,0))
        self.gy = _add(polyder(self.p,axis=1),-2*_shift(self.p,1))
        self.jxx=polyder(self.gx,axis=0); self.jxy=polyder(self.gx,axis=1)
        self.jyx=polyder(self.gy,axis=0); self.jyy=polyder(self.gy,axis=1)
        off=0 if subsystem=='cavity' else 2
        self.mu=np.sqrt(2)*evolution.displacement(time)[off:off+2].real

    def gradient_polynomial(self,z):
        x,y=z
        return np.array([polyval2d(x,y,self.gx),polyval2d(x,y,self.gy)])

    def jacobian(self,z):
        x,y=z
        return np.array([[polyval2d(x,y,self.jxx),polyval2d(x,y,self.jxy)],
                         [polyval2d(x,y,self.jyx),polyval2d(x,y,self.jyy)]])

    def density(self,z):
        return float(np.exp(-np.dot(z,z))*polyval2d(*z,self.p))

    def minima(self, guesses, radius=3.):
        candidates=[]
        for guess in guesses:
            answer=root(self.gradient_polynomial,guess,jac=self.jacobian,tol=1e-10)
            z=answer.x
            if not np.all(np.isfinite(z)) or np.linalg.norm(z)>5:
                continue
            if np.linalg.norm(self.gradient_polynomial(z))>1e-8:
                continue
            position=z+self.mu
            if np.linalg.norm(position)>radius:
                continue
            hessian=self.jacobian(z)*np.exp(-np.dot(z,z))
            # At a critical point the Jacobian of G times exp(-z^2)
            # is the Hessian of the full density, including its envelope.
            if np.linalg.eigvalsh((hessian+hessian.T)/2).min()<=1e-8:
                continue
            if all(np.linalg.norm(position-other)>1e-5 for other in candidates):
                candidates.append(position)
        return candidates


def track_density_minimum(evolution, times, subsystem, radius=3., max_step=.5):
    p=evolution.physical
    initial=np.array([getattr(p,subsystem+'_x'),getattr(p,subsystem+'_y')])/p.oscillator_width
    previous=initial.copy()
    track=np.full((len(times),2),np.nan)
    was_missing=True
    for i,t in enumerate(times):
        profile=DensityProfile(evolution,float(t),subsystem)
        guesses=[previous-profile.mu,initial-profile.mu,np.zeros(2)]
        # Multiple seeds recover branches after a minimum disappears/reappears.
        guesses += [np.array([x,y]) for x in [-1.5,0,1.5] for y in [-1.5,0,1.5]]
        candidates=profile.minima(guesses,radius)
        if not candidates:
            was_missing=True
            continue
        point=min(candidates,key=lambda q: np.linalg.norm(q-previous))
        jump=np.linalg.norm(point-previous)
        if i==0 or jump<=max_step or was_missing:
            track[i]=point
        previous=point
        was_missing=False
    return track
