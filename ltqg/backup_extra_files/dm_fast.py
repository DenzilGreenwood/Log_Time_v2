# dm_fast.py
from dataclasses import dataclass
import numpy as np
from typing import Optional
from numpy.typing import ArrayLike
from math import log10
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize
import argparse

# ---------------------------
# Constants
# ---------------------------
C_KM_S = 299792.458

@dataclass
class CosmoParams:
    H0: float = 70.0
    Omega_m: float = 0.3
    w0: float = -1.0
    wa: float = 0.0
    Omega_r: float = 0.0

class Cosmology:
    def __init__(self, p: CosmoParams):
        self.p = p
        self.Om = p.Omega_m
        self.Or = p.Omega_r
        self.w0 = p.w0
        self.wa = p.wa
        self.Ode = 1.0 - self.Om - self.Or

    def E(self, z: ArrayLike) -> ArrayLike:
        z = np.asarray(z)
        f_m  = self.Om * (1.0 + z)**3
        f_r  = self.Or * (1.0 + z)**4
        f_de = self.Ode * (1.0 + z)**(3.0*(1.0 + self.w0 + self.wa)) * np.exp(-3.0*self.wa * z/(1.0+z))
        return np.sqrt(f_m + f_r + f_de)

    # ---- Standard distances
    def D_C_std(self, z: float) -> float:
        integral, _ = quad(lambda zz: 1.0/self.E(zz), 0.0, z, epsabs=1e-10, epsrel=1e-10, limit=300)
        return (C_KM_S/self.p.H0) * integral

    def D_L_std(self, z: float) -> float:
        return (1.0 + z) * self.D_C_std(z)

    # ---- σ-grid distances (LTQG)
    def _sigma_system(self, sigma: float, y: np.ndarray) -> np.ndarray:
        z, X = y
        factor = (1.0 + z) * np.exp(sigma)
        dz_dsigma = - factor * self.E(z)
        dX_dsigma = - factor
        return np.array([dz_dsigma, dX_dsigma], dtype=float)

    def D_C_sigma(self, z_target: float, sigma_min: float=-50.0) -> float:
        def event_hit(s, y): return y[0] - z_target
        event_hit.terminal, event_hit.direction = True, 1.0
        sol = solve_ivp(self._sigma_system, (0.0, sigma_min), np.array([0.0, 0.0]),
                        events=event_hit, atol=1e-12, rtol=1e-10, max_step=0.05)
        if sol.status == 1 and sol.t_events[0].size:
            z_hit, X_hit = sol.y_events[0][0]
        else:
            z_hit, X_hit = sol.y[:, -1]
        return (C_KM_S/self.p.H0) * X_hit

    def D_L_sigma(self, z: float) -> float:
        return (1.0 + z) * self.D_C_sigma(z)

# ---------------------------
# Synthetic SN dataset
# ---------------------------
@dataclass
class SNDataset:
    z: np.ndarray
    mu: np.ndarray
    sigma_mu: np.ndarray

    @staticmethod
    def synthetic(n: int=40, seed: int=42, cosmo: Optional[Cosmology]=None, M_true: float=-19.3):
        rng = np.random.default_rng(seed)
        if cosmo is None:
            cosmo = Cosmology(CosmoParams())
        z = np.sort(rng.uniform(0.05, 0.9, size=n))
        Dl = np.array([cosmo.D_L_std(zi) for zi in z])
        mu_true = 5*np.log10(Dl) + 25 + M_true
        sigma_mu = 0.12*np.ones_like(z)
        mu = mu_true + rng.normal(0, sigma_mu)
        return SNDataset(z=z, mu=mu, sigma_mu=sigma_mu)

def mu_theory(z: ArrayLike, cosmo: Cosmology, use_sigma: bool, M: float) -> np.ndarray:
    if use_sigma:
        Dl = np.array([cosmo.D_L_sigma(zi) for zi in z])
    else:
        Dl = np.array([cosmo.D_L_std(zi) for zi in z])
    return 5.0*np.log10(Dl) + 25.0 + M

def sn_negloglike(theta: np.ndarray, data: SNDataset, use_sigma: bool) -> float:
    H0, Om, M = theta
    # box priors
    if not (40 < H0 < 95 and 0.05 < Om < 0.6): return 1e50
    cosmo = Cosmology(CosmoParams(H0=H0, Omega_m=Om))
    mu_th = mu_theory(data.z, cosmo, use_sigma, M)
    chi2 = np.sum(((data.mu - mu_th)/data.sigma_mu)**2)
    return 0.5*chi2

def self_check():
    cosmo = Cosmology(CosmoParams())
    zs = np.linspace(1e-4, 2.0, 40)
    Dl_std = np.array([cosmo.D_L_std(z) for z in zs])
    Dl_sig = np.array([cosmo.D_L_sigma(z) for z in zs])
    rel = np.max(np.abs(Dl_std - Dl_sig)/Dl_std)
    print(f"[Self-check] Max rel. error std vs σ: {rel:.3e}")
    return rel

def run_fit(use_sigma: bool, do_plot: bool):
    print("LTQG Ω_Λ Fit Demo (SN-only, synthetic)")
    print("======================================")
    err = self_check()
    if err < 5e-4: print("OK: σ-integrator matches standard distances.\n")
    else:          print("Warning: larger mismatch than expected.\n")

    # Data
    truth = Cosmology(CosmoParams(H0=70.0, Omega_m=0.30))
    data = SNDataset.synthetic(40, 42, truth, -19.3)
    print(f"SNe: {len(data.z)} objects, z in [{data.z.min():.3f}, {data.z.max():.3f}]")

    # Fit
    x0 = np.array([70.0, 0.3, -19.3])
    res = minimize(lambda th: sn_negloglike(th, data, use_sigma),
                   x0, method="Nelder-Mead",
                   options={"xatol":1e-4, "fatol":1e-4, "maxiter":2000, "disp": False})
    H0, Om, M = res.x
    Ol = 1.0 - Om
    print(("\nFit ({})".format("σ-integrator" if use_sigma else "standard")))
    print("---------------")
    print(f"H0  = {H0:.2f} km/s/Mpc")
    print(f"Ωm  = {Om:.3f}")
    print(f"ΩΛ  = {Ol:.3f}")
    print(f"M   = {M:.3f}")
    print(f"χ²/DoF ≈ {2*res.fun:.2f}/{len(data.z)-3}")

    if do_plot:
        import matplotlib.pyplot as plt
        cosmo = Cosmology(CosmoParams(H0=H0, Omega_m=Om))
        mu_th = mu_theory(data.z, cosmo, use_sigma, M)
        plt.figure()
        plt.errorbar(data.z, data.mu - mu_th, yerr=data.sigma_mu, fmt='o', ms=4)
        plt.axhline(0, ls='--')
        plt.xlabel('z'); plt.ylabel('μ_obs − μ_th')
        plt.title('SN residuals ({})'.format('σ' if use_sigma else 'standard'))
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-sigma", action="store_true", help="use LTQG σ-integrator")
    ap.add_argument("--plot", action="store_true", help="plot residuals")
    args = ap.parse_args()
    run_fit(use_sigma=args.use_sigma, do_plot=args.plot)
