"""
LTQG Ω_Λ Inference Demo (σ-uniform integrator vs standard z-integrator)
======================================================================

What this gives you
-------------------
1) Cosmology class for flat ΛCDM and w0–waCDM.
2) Distances via standard z-integrals.
3) Distances via an LTQG σ-grid ODE:
      σ := log(t/t0),  choose t0 := 1/H0  => (H0 t0 = 1)
   E(z) := H(z)/H0
   Coupled system in σ:
      dz/dσ = - (1+z) * e^σ * E(z)
      dX/dσ = - (1+z) * e^σ
   where X(σ) = D_C(σ) * H0 / c  is the *dimensionless* comoving distance.
   Integrate from σ=0 (today, z=0, X=0) backwards (σ<0) until z reaches target z_s.
4) SN likelihood for distance moduli µ = 5 log10(D_L/Mpc) + 25 with a nuisance M.
5) A tiny MCMC (if `emcee` is installed); otherwise a grid scan.

How to replace the synthetic data with real data later
------------------------------------------------------
- SN (Pantheon+): CSV with columns (z, mu, sigma_mu). Load and pass to `SNDataset`.
- BAO: Add a BAO likelihood (DV/rd, DA/rd, H*rd) with covariances (DESI, BOSS).
- CMB distance priors: Add shift-parameter likelihood (R, lA, Ω_b h^2).

Dependencies
------------
- numpy, scipy
- emcee (optional)
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, Optional, Callable
from math import log10
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize

# Try to import emcee (optional)
try:
    import emcee
    HAS_EMCEE = True
except Exception:
    HAS_EMCEE = False


# ---------------------------
# Constants and unit helpers
# ---------------------------
C_KM_S   = 299792.458          # speed of light [km/s]
MPC_IN_KM = 3.0856775814913673e19 / 1000.0  # Mpc in km (exact-ish), not strictly needed


@dataclass
class CosmoParams:
    H0: float = 70.0           # H0 in km/s/Mpc
    Omega_m: float = 0.3       # matter density
    w0: float = -1.0           # dark-energy EOS at z=0
    wa: float = 0.0            # EOS evolution parameter (CPL)
    # flatness: Ω_k = 0 -> Ω_de = 1 - Ω_m - Ω_r  (we neglect Ω_r by default here)
    Omega_r: float = 0.0       # radiation fraction (set small if needed, else 0)


class Cosmology:
    """
    Flat w0–waCDM (with optional Ω_r). ΛCDM is w0=-1, wa=0.
    """
    def __init__(self, params: CosmoParams):
        self.p = params
        self.Om = self.p.Omega_m
        self.Or = self.p.Omega_r
        self.w0 = self.p.w0
        self.wa = self.p.wa
        self.Ode = 1.0 - self.Om - self.Or

    def E(self, z: ArrayLike) -> ArrayLike:
        """
        E(z) = H(z)/H0
        w(a) = w0 + wa(1-a) with a = 1/(1+z)
        ρ_de ∝ a^{-3(1+w0+wa)} exp[-3 wa (1-a)]
        => f_de(z) = (1+z)^{3(1+w0+wa)} * exp(-3 wa z/(1+z))
        """
        z = np.asarray(z)
        f_m  = self.Om * (1.0 + z)**3
        f_r  = self.Or * (1.0 + z)**4
        f_de = self.Ode * (1.0 + z)**(3.0*(1.0 + self.w0 + self.wa)) * np.exp(-3.0*self.wa * z/(1.0+z))
        return np.sqrt(f_m + f_r + f_de)

    # -------------------------------
    # Standard distances (z-integral)
    # -------------------------------
    def D_C_std(self, z: float) -> float:
        """
        Line-of-sight comoving distance [Mpc]
        D_C = (c/H0) ∫_0^z dz'/E(z')
        """
        integral, _ = quad(lambda zz: 1.0/self.E(zz), 0.0, z, epsabs=1e-10, epsrel=1e-10, limit=500)
        return (C_KM_S/self.p.H0) * integral

    def D_L_std(self, z: float) -> float:
        """Luminosity distance [Mpc] for flat cosmology: D_L = (1+z) D_C."""
        return (1.0 + z) * self.D_C_std(z)

    # ---------------------------------------------
    # LTQG σ-grid distances (reparameterized ODEs)
    # ---------------------------------------------
    def _sigma_system(self, sigma: float, y: np.ndarray) -> np.ndarray:
        """
        ODE in σ = log(t/t0), choosing t0 := 1/H0 (so H0 t0 = 1).
        State vector: y = [z(σ), X(σ)]
           z' = dz/dσ = - (1+z) * e^σ * E(z)
           X' = dX/dσ = - (1+z) * e^σ
        where X = D_C * H0 / c  (dimensionless comoving distance).
        """
        z, X = y
        factor = (1.0 + z) * np.exp(sigma)
        dz_dsigma = - factor * self.E(z)
        dX_dsigma = - factor
        return np.array([dz_dsigma, dX_dsigma], dtype=float)

    def D_C_sigma(self, z_target: float, sigma_min: float=-50.0) -> float:
        """
        Compute D_C via σ-stepping.
        Start at σ=0 with (z=0, X=0), integrate *backwards* until z reaches z_target.
        We stop early if z hits z_target before sigma_min.
        Returns D_C in Mpc.
        """
        # Event: stop when z - z_target = 0 (crossing from below)
        def event_reach_z(sigma, y):
            return y[0] - z_target
        event_reach_z.terminal = True
        event_reach_z.direction = 1.0  # we integrate backwards; z increases as σ decreases

        sol = solve_ivp(
            fun=self._sigma_system,
            t_span=(0.0, sigma_min),   # integrate from 0 downward to sigma_min
            y0=np.array([0.0, 0.0], dtype=float),
            events=event_reach_z,
            atol=1e-12, rtol=1e-10, max_step=0.05
        )
        if sol.status == 1 and sol.t_events[0].size > 0:
            # Interpolate X at the event
            sigma_hit = sol.t_events[0][0]
            z_hit, X_hit = sol.y_events[0][0]
        else:
            # If we didn't hit (very high z_target), take the last point
            z_hit, X_hit = sol.y[:, -1]

        # Dimensionalize: D_C = (c/H0) * X
        return (C_KM_S/self.p.H0) * X_hit

    def D_L_sigma(self, z: float) -> float:
        """Luminosity distance [Mpc] from σ-stepping."""
        return (1.0 + z) * self.D_C_sigma(z)


# -------------------------
# Supernova (SN) likelihood
# -------------------------
@dataclass
class SNDataset:
    z: np.ndarray
    mu: np.ndarray
    sigma_mu: np.ndarray

    @staticmethod
    def synthetic(n: int=25, rng_seed: int=7, cosmology: Optional[Cosmology]=None, M_true: float=-19.3) -> "SNDataset":
        """
        Make a tiny synthetic sample for demo. z in [0.01, 0.9].
        """
        rng = np.random.default_rng(rng_seed)
        if cosmology is None:
            cosmology = Cosmology(CosmoParams())

        z = np.sort(rng.uniform(0.01, 0.9, size=n))
        D_L = np.array([cosmology.D_L_std(zi) for zi in z])  # use standard for ground truth
        mu_true = 5.0*np.log10(D_L) + 25.0 + M_true
        sigma_mu = 0.12*np.ones_like(z)
        mu_obs = mu_true + rng.normal(0.0, sigma_mu)
        return SNDataset(z=z, mu=mu_obs, sigma_mu=sigma_mu)


def mu_theory(z: ArrayLike, cosmo: Cosmology, use_sigma: bool=False, M: float=-19.3) -> np.ndarray:
    """Distance modulus prediction array for a list of redshifts."""
    z = np.asarray(z)
    if use_sigma:
        Dl = np.array([cosmo.D_L_sigma(zi) for zi in z])
    else:
        Dl = np.array([cosmo.D_L_std(zi) for zi in z])
    return 5.0*np.log10(Dl) + 25.0 + M


def sn_loglike(params_vec: np.ndarray, data: SNDataset, use_sigma: bool=False,
               model: str="lcdm") -> float:
    """
    params_vec:
        model=="lcdm"        -> [H0, Omega_m, M]
        model=="w0wa"        -> [H0, Omega_m, w0, wa, M]
    Returns log-likelihood (Gaussian in µ).
    """
    if model == "lcdm":
        H0, Om, M = params_vec
        cosmo = Cosmology(CosmoParams(H0=H0, Omega_m=Om, w0=-1.0, wa=0.0))
    elif model == "w0wa":
        H0, Om, w0, wa, M = params_vec
        cosmo = Cosmology(CosmoParams(H0=H0, Omega_m=Om, w0=w0, wa=wa))
    else:
        raise ValueError("Unknown model")

    mu_th = mu_theory(data.z, cosmo, use_sigma=use_sigma, M=M)
    chi2 = np.sum(((data.mu - mu_th)/data.sigma_mu)**2)
    # flat priors in a sensible box; return -0.5 chi^2 if inside, -inf otherwise
    if not (40.0 < H0 < 95.0 and 0.05 < cosmo.Om < 0.6):
        return -np.inf
    if model == "w0wa":
        if not (-2.5 < w0 < -0.3 and -3.0 < wa < 3.0):
            return -np.inf
    return -0.5*chi2


# -----------------------
# Simple fit utilities
# -----------------------
def run_grid_search_sn(data: SNDataset, use_sigma: bool=False, model: str="lcdm"):
    """
    Coarse grid search (fast, deterministic) to demonstrate that σ and standard match.
    """
    if model == "lcdm":
        H0_grid = np.linspace(60.0, 80.0, 41)    # km/s/Mpc
        Om_grid = np.linspace(0.15, 0.45, 41)
        M_grid  = np.linspace(-19.6, -19.0, 31)
        best = (-np.inf, None)
        for H0 in H0_grid:
            for Om in Om_grid:
                for M in M_grid:
                    ll = sn_loglike(np.array([H0, Om, M]), data, use_sigma=use_sigma, model=model)
                    if ll > best[0]:
                        best = (ll, (H0, Om, M))
        return best
    else:
        raise NotImplementedError("Grid for w0–wa omitted for brevity.")


def run_emcee_sn(data: SNDataset, use_sigma: bool=False, model: str="lcdm",
                 nwalkers: int=24, nsteps: int=2000, burn: int=500, seed: int=7):
    """
    Tiny emcee run (only if emcee is installed). Returns samples and log-prob function.
    """
    assert HAS_EMCEE, "emcee not available"
    rng = np.random.default_rng(seed)

    if model == "lcdm":
        def log_prob(theta):
            return sn_loglike(theta, data, use_sigma=use_sigma, model=model)
        # Initialize walkers around a plausible center:
        p0_center = np.array([70.0, 0.3, -19.3])
        p0 = p0_center + rng.normal(0, [2.0, 0.05, 0.05], size=(nwalkers, 3))
        ndim = 3
    else:
        raise NotImplementedError("MCMC for w0–wa omitted for brevity.")

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, nsteps, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)
    return chain, log_prob


# -----------------------
# Demo / self-check main
# -----------------------
def self_check_distances():
    """
    Show D_L(z) agreement between standard and LTQG σ-integrator over a grid.
    Prints max relative error.
    """
    cosmo = Cosmology(CosmoParams(H0=70.0, Omega_m=0.3, w0=-1.0, wa=0.0))
    zs = np.linspace(1e-4, 2.0, 50)
    Dl_std = np.array([cosmo.D_L_std(z) for z in zs])
    Dl_sig = np.array([cosmo.D_L_sigma(z) for z in zs])
    rel_err = np.max(np.abs(Dl_std - Dl_sig)/Dl_std)
    print(f"[Self-check] Max rel. error between standard and σ-integrator: {rel_err:.3e}")
    return rel_err


def main():
    print("LTQG Ω_Λ Fit Demo (SN-only, synthetic)")
    print("======================================")
    # 0) Sanity: verify σ-integrator matches standard
    err = self_check_distances()
    if err > 5e-4:
        print("WARNING: relative error is higher than expected; consider tighter tolerances.")
    else:
        print("OK: σ-integrator matches standard distances.")

    # 1) Build synthetic SN dataset (uses standard cosmology for truth)
    true_cosmo = Cosmology(CosmoParams(H0=70.0, Omega_m=0.3))
    data = SNDataset.synthetic(n=40, rng_seed=42, cosmology=true_cosmo, M_true=-19.3)
    print(f"SNe: {len(data.z)} objects, z in [{data.z.min():.3f}, {data.z.max():.3f}]")

    # 2) Fit with standard distances
    best_std = run_grid_search_sn(data, use_sigma=False, model="lcdm")
    ll_std, (H0_s, Om_s, M_s) = best_std
    print(f"Standard fit (grid): H0={H0_s:.2f}, Ωm={Om_s:.3f}, M={M_s:.3f}, lnL={ll_std:.2f}")

    # 3) Fit with σ-integrator distances
    best_sig = run_grid_search_sn(data, use_sigma=True, model="lcdm")
    ll_sig, (H0_q, Om_q, M_q) = best_sig
    print(f"σ-integrator fit (grid): H0={H0_q:.2f}, Ωm={Om_q:.3f}, M={M_q:.3f}, lnL={ll_sig:.2f}")

    # 4) Optional: emcee demonstration
    if HAS_EMCEE:
        print("\nRunning a tiny emcee chain (standard distances)...")
        chain_std, _ = run_emcee_sn(data, use_sigma=False, model="lcdm", nwalkers=24, nsteps=2000, burn=500)
        means_std = np.mean(chain_std, axis=0)
        print(f"emcee (standard) posterior mean ~ H0={means_std[0]:.2f}, Ωm={means_std[1]:.3f}, M={means_std[2]:.3f}")

        print("Running a tiny emcee chain (σ-integrator distances)...")
        chain_sig, _ = run_emcee_sn(data, use_sigma=True, model="lcdm", nwalkers=24, nsteps=2000, burn=500)
        means_sig = np.mean(chain_sig, axis=0)
        print(f"emcee (σ) posterior mean ~ H0={means_sig[0]:.2f}, Ωm={means_sig[1]:.3f}, M={means_sig[2]:.3f}")
    else:
        print("\nTip: install `emcee` to run a quick MCMC:  pip install emcee")

    print("\nInterpretation")
    print("--------------")
    print("If the σ-integrator fit matches the standard fit (they should within numerical noise),")
    print("you’ve validated that reclocking to σ preserves the dark-energy inference (Ω_Λ=1-Ω_m).")
    print("Replace the synthetic SNe with Pantheon+ and add BAO/CMB priors to get real constraints.")


if __name__ == "__main__":
    main()
