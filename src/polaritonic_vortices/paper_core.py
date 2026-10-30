"""Paper-inspired core prescription, retaining the exact excitation content.

The paper minimizes the factor left after removing a fixed Gaussian.
Here that factor is evaluated exactly, without a quadratic Taylor truncation.
For coherent backgrounds it is exponential times quartic, not just quartic.
Accepted points are local minima of this factor, not topological certificates.
"""
import numpy as np
from numpy.polynomial.polynomial import polyval2d, polyder
from scipy.optimize import root
from .density_minima import DensityProfile, _add


class GaussianRemovedProfile:
    def __init__(self, evolution, time, subsystem, envelope="fixed"):
        if envelope not in {"fixed", "coherent"}:
            raise ValueError("envelope must be fixed or coherent")
        self.profile = DensityProfile(evolution, time, subsystem)
        self.mu = self.profile.mu
        self.p = self.profile.p
        self.envelope = envelope
        self.slope = 2*self.mu if envelope == "fixed" else np.zeros(2)
        self.g = [_add(polyder(self.p, axis=i), self.slope[i]*self.p)
                  for i in range(2)]
        self.j = [[polyder(self.g[i], axis=j) for j in range(2)] for i in range(2)]
        self.scale = max(1., np.max(np.abs(self.p)))

    def gradient_factor(self, z):
        return np.array([polyval2d(*z, q) for q in self.g])

    def jacobian(self, z):
        return np.array([[polyval2d(*z, q) for q in row] for row in self.j])

    def value(self, position):
        z=np.asarray(position)-self.mu
        exponent=np.dot(self.slope,z)+(np.dot(self.mu,self.mu) if self.envelope=="fixed" else 0.)
        return float(np.exp(exponent)*polyval2d(*z,self.p))

    def minima(self, guesses, radius=12., tolerance=1e-8):
        candidates=[]
        for guess in guesses:
            sol=root(self.gradient_factor, guess, jac=self.jacobian, tol=1e-10)
            z=sol.x
            if not np.isfinite(z).all() or np.linalg.norm(z+self.mu)>radius:
                continue
            residual=np.linalg.norm(self.gradient_factor(z))/self.scale
            if residual>tolerance:
                continue
            # At a stationary point, the positive exponential factor can be
            # omitted when checking the Hessian's sign.
            jac=self.jacobian(z)
            hessian=(jac+jac.T)/2
            eigen=np.linalg.eigvalsh(hessian)
            if eigen[0] <= 1e-9*max(1.,np.linalg.norm(hessian,2)):
                continue
            position=z+self.mu
            if all(np.linalg.norm(position-c[0])>1e-6 for c in candidates):
                candidates.append((position,residual,float(eigen[0])))
        return candidates


def track_paper_core(evolution, times, subsystem, *, envelope="fixed", radius=12., max_step=.5):
    """Track a continuous local minimum; rejected points remain NaN.

    Coordinates and radius are in units of w in the documented common frame.
    radius is a numerical search bound, not an excitation cutoff. Raw accepted
    locations, candidate counts, residuals and branch breaks are returned.
    """
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("times must be finite and strictly increasing")
    if subsystem not in {"cavity","exciton"} or radius<=0 or max_step<=0:
        raise ValueError("Invalid subsystem or search settings")
    p=evolution.physical
    initial=np.array([getattr(p,subsystem+'_x'),getattr(p,subsystem+'_y')])/p.oscillator_width
    previous=initial.copy(); missing=True
    xy=np.full((len(times),2),np.nan); residual=np.full(len(times),np.nan)
    hessian=np.full(len(times),np.nan); counts=np.zeros(len(times),int)
    breaks=np.zeros(len(times),bool)
    for i,t in enumerate(times):
        profile=GaussianRemovedProfile(evolution,t,subsystem,envelope)
        guesses=[previous-profile.mu,initial-profile.mu,np.zeros(2)]
        guesses += [np.array([x,y]) for x in [-3.,-1.5,0.,1.5,3.] for y in [-3.,-1.5,0.,1.5,3.]]
        candidates=profile.minima(guesses,radius)
        counts[i]=len(candidates)
        if not candidates:
            missing=True; continue
        point,err,eig=min(candidates,key=lambda c:np.linalg.norm(c[0]-previous))
        jump=np.linalg.norm(point-previous)>max_step
        if i==0 or missing or not jump:
            xy[i]=point; residual[i]=err; hessian[i]=eig
            breaks[i]=missing or jump
        else:
            missing=True
            previous=point
            continue
        previous=point;missing=False
    return dict(time_ps=times,xy_over_w=xy,gradient_residual=residual,
                hessian_min=hessian,candidate_count=counts,branch_break=breaks)
