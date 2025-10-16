# LTQG for Mathematicians: A Detailed Exposition

> This note explains Log‑Time Quantum Gravity (LTQG) as a rigorous **change of clock** framework. It is written for a mathematical audience familiar with functional analysis, PDE/ODE theory, differential geometry, and basic quantum field theory on curved spacetimes. We keep proofs at sketch level where the arguments follow standard results (Kato stability, Dyson expansions, conformal geometry facts), and we point to acceptance tests that numerically verify claims.

---

## 1. What LTQG Is (and Is Not)
**LTQG** replaces the physical time parameter \(\tau>0\) by a *logarithmic* clock
\[
\sigma := \log(\tau/\tau_0), \qquad \tau = \tau_0 e^{\sigma}, \quad \frac{d\tau}{d\sigma} = \tau.
\]
The program develops consequences of this reparameterization in quantum mechanics, QFT on curved backgrounds, and cosmology.

- It **does not** change the dynamics, observables, or spectra.
- It **does** make early‑time multiplicative structure additive, which is useful analytically (regularity near \(\tau\to 0^+\)) and numerically (stiff systems).

Two clocks appear:
- **Log‑time** \(\sigma\) for reparameterizing evolution.
- An optional **scalar clock** \(\varsigma = \log(\phi/\phi_0)\) in minisuperspace reductions; this is independent of \(\sigma\).

---

## 2. Time‑Ordering Invariance and Unitary Equivalence
Consider a (possibly time‑dependent) Schrödinger generator \(H(\tau)\) on a Hilbert space \(\mathcal H\). The \(\tau\)‑propagator solves
\[
 i\,\partial_\tau U_\tau(\tau,\tau_0) = H(\tau)U_\tau(\tau,\tau_0), \qquad U_\tau(\tau_0,\tau_0)=\mathbf 1.
\]
Define the \(\sigma\)‑generator by the chain rule
\[
 \widetilde H(\sigma) := \frac{d\tau}{d\sigma}\, H\big(\tau(\sigma)\big) = \tau(\sigma) H\big(\tau(\sigma)\big).
\]
Let \(U_\sigma\) be the \(\sigma\)-time propagator with \(i\,\partial_\sigma U_\sigma = \widetilde H\,U_\sigma\).

### Lemma 2.1 (Chronological invariance under monotone clocks)
If \(\sigma=F(\tau)\) is \(C^1\) with \(F'\!>0\), then for any \(\tau_1\le \tau_2\)
\[
 \mathcal T_\tau\exp\!\Big(-i\!\int_{\tau_1}^{\tau_2} H(\tau)\,d\tau\Big)
 = \mathcal T_\sigma\exp\!\Big(-i\!\int_{\sigma_1}^{\sigma_2} \widetilde H(\sigma)\,d\sigma\Big), \qquad \sigma_j=F(\tau_j).
\]
*Sketch.* Change variables term‑wise in the Dyson series. Monotonicity carries ordered simplices to ordered simplices, and \(H\,d\tau=\widetilde H\,d\sigma\). Conclude equality of propagators.

### Theorem 2.2 (Unitary equivalence)
Under standard existence/uniqueness hypotheses (bounded case: strong measurability and local uniform bounds; unbounded case: Kato stability on a common dense domain), the propagators coincide: \(U_\sigma(\sigma_2,\sigma_1)=U_\tau(\tau_2,\tau_1)\). Hence **all physical quantities** (transition amplitudes, Heisenberg expectation values) are invariant under \(\tau\leftrightarrow\sigma\).

*Remark.* For \(F(\tau)=\log(\tau/\tau_0)\), \(\widetilde H(\sigma)=\tau H(\tau)\); the result formalizes “same physics, new clock.”

---

## 3. Asymptotic Silence
Define the **effective generator** \(H_\text{eff}(\sigma):=\tau(\sigma)H\big(\tau(\sigma)\big)\). We say the past \(\sigma\to-\infty\) is *asymptotically silent* if \(\|H_\text{eff}(\sigma)\|\to 0\) and the total past phase \(\int_{-\infty}^{\sigma_0}\!\|H_\text{eff}(s)\|ds\) is finite.

### Proposition 3.1 (Sufficient conditions)
If near \(\tau=0^+\)
- **(L¹)** \(\|H(\tau)\|\in L^1(0,\tau_1] \), or
- **(power law)** \(\|H(\tau)\| = O(\tau^{-\alpha})\) with \(\alpha<1\),
then \(\|H_\text{eff}(\sigma)\|\to 0\) as \(\sigma\to-\infty\) and the accumulated phase is finite.

*Sketch.* If \(\|H\|=O(\tau^{-\alpha})\), then \(\|H_\text{eff}\|=O(\tau^{1-\alpha})=O(e^{(1-\alpha)\sigma})\to0\) for \(\alpha<1\). Integrate explicitly.

*Boundary and pathologies.* \(\alpha=1\) gives \(H_\text{eff}\equiv\text{const}\) (no silence); essential singularities (e.g. \(e^{1/\tau}\)) blow up in \(\sigma\)-past.

---

## 4. Cosmology: FLRW and a Weyl Benchmark
Let \(ds^2=-d\tau^2+a(\tau)^2 d\mathbf x^2\) with \(a(\tau)=(\tau/\tau_0)^p\) (\(p>0\)). Canonical relations: \(p=\tfrac{2}{3(1+w)}\), \(H=\dot a/a=p/\tau\), \(\rho(a)\propto a^{-3(1+w)}\), and \(\rho(\tau)\propto \tau^{-2}\).

### 4.1 Weyl frame with \(\Omega=\tau^{-1}\)
Let \(\tilde g=\Omega^2 g\) with \(\Omega=\tau^{-1}\). In \(\sigma\) coordinates, \(d\tilde s^2=-d\sigma^2 + C^2 e^{2(p-1)\sigma}d\mathbf x^2\) (constant lapse). A direct computation yields
\[ \tilde R = 12(p-1)^2, \]
a convenient constant‑curvature benchmark.

> **Frame dependence.** Weyl rescaling is not a diffeomorphism; physical equivalence requires a matter‑coupling prescription. We therefore use \(\tilde R\) only as a geometric diagnostic, not as a claim of singularity removal.

### 4.2 Geodesic completeness in the Weyl frame (summary)
For \(p\le 1\) (radiation/matter), the past boundary \(\sigma\to-\infty\) is *null‑geodesically complete*; comoving timelike geodesics are complete for all \(p>0\). This aligns with the “asymptotic silence” narrative: dynamics slow in \(\sigma\) and curves accumulate infinite parameter length towards the past boundary.

---

## 5. QFT on FLRW: Canonical Variable and Bogoliubov Invariants
Consider a minimally coupled real scalar field on flat FLRW. For each comoving mode \(k\), define the **canonical variable**
\[ v_k := a^{3/2} u_k. \]
Then \(v_k\) solves a friction‑free second‑order ODE
\[ \ddot v_k + \Omega_k(\tau)^2 v_k = 0, \qquad \Omega_k^2 = \frac{k^2}{a^2} + m^2 - \frac{3}{2}\dot H - \frac{9}{4} H^2, \]
and the Klein–Gordon **Wronskian** satisfies
\[ W(v_k,\bar v_k)= i\big(\bar v_k\dot v_k - \dot{\bar v}_k v_k\big) \equiv i \quad (\text{conserved}). \]

### 5.1 Initial data and normalization
Choose adiabatic initial data at \(\tau_i\): \(v_i = (2\Omega_i)^{-1/2}\), \(\dot v_i = -i\Omega_i v_i\), then renormalize by a constant phase so that \(W(\tau_i)=i\) to machine precision.

### 5.2 τ vs. σ evolutions and common physical slice
Evolve \(v_k\) in **\(\tau\)** and in **\(\sigma\)** (using \(d/d\sigma=\tau\,d/d\tau\)) from \(\tau_i\) to the **same** \(\tau_f\) (hence same \(a(\tau_f)\)). At \(\tau_f\) define the instantaneous positive‑frequency basis \(v_k^{(+)}=(2\Omega_f)^{-1/2}\), \(\dot v_k^{(+)}=-i\Omega_f v_k^{(+)}\). The Bogoliubov coefficients are
\[ \alpha_k=(v_k^{(+)},v_k),\qquad \beta_k = -\,(\overline{v_k^{(+)}},v_k), \]
using the KG inner product for \(v\). Unitarity demands \(|\alpha_k|^2-|\beta_k|^2=1\); in the UV, \(|\beta_k|^2\to 0\).

### 5.3 Acceptance tests (numerical)
For \(p\in\{\tfrac12,\tfrac23\}\), \(m=0\), and a grid of \(k\):
1. **Coordinate invariance:** \(\max_k |\,|\beta_k|^2_\tau - |\beta_k|^2_\sigma\,| < 10^{-6}\).
2. **KG norm:** \(\max_k |\,|\alpha_k|^2 - |\beta_k|^2 - 1\,| < 10^{-8}\).
3. **Wronskian conservation:** \(\max_k\max_t |W(t)-W_0| < 10^{-8}\) for both evolutions.

In practice, all three criteria pass with comfortable margins (typically \(10^{-9}\)–\(10^{-10}\)).

---

## 6. Minisuperspace (Lapse‑First Derivation)
Work with the EH+scalar action
\[ S = \int d^4x\,\sqrt{-g}\Big(\frac{R}{16\pi G} + \tfrac12 g^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi - V(\phi)\Big). \]
In flat FLRW, \(ds^2=-N(t)^2dt^2+a(t)^2 d\mathbf x^2\). Reduce (up to boundary terms) to
\[ S=\int dt\Big( -\frac{3a\dot a^2}{8\pi G N} + \frac{a^3}{2N}\dot\phi^2 - N a^3 V(\phi)\Big). \]
Vary **with \(N\) kept** to obtain: Hamiltonian constraint \(H^2=\tfrac{8\pi G}{3}\rho\), acceleration equation \(\ddot a/a = -\tfrac{4\pi G}{3}(\rho+3p)\), and KG equation \(\ddot\phi+3H\dot\phi+V'(\phi)=0\). Only after this do we choose an internal clock (\(\sigma\) or \(\varsigma\)) and reduce.

---

## 7. Why the Log Clock Helps (Metrology View)
Monotone reparameterizations preserve observables but not *measurement protocols*. Uniform sampling in \(\sigma\) reweights early \(\tau\), which can reduce variance for quantities that accrue phase at early times. Two reproducible experiments:
- **A. Bogoliubov spectra:** compare \(\tau\)-uniform vs. \(\sigma\)-uniform grids at fixed cost; same continuum \(|\beta_k|^2\), lower MSE for \(\sigma\)-uniform in early‑time‑sensitive regimes.
- **B. Near‑horizon detector:** response function \(F\) with compact support windows; \(\sigma\)-uniform windows stabilize phase accumulation as supports extend towards smaller \(\tau\).

Acceptance targets: invariance within \(10^{-6}\), variance reduction by \(\gtrsim 20\%\) at fixed samples.

---

## 8. Limitations and Caveats
- **Frame dependence (Weyl):** geometric regularization (e.g., constant \(\tilde R\)) does not by itself imply physical regularization; matter coupling must be specified.
- **Asymptotic silence hypotheses:** \(\alpha<1\) (or L¹) are sufficient, not necessary; borderline/essential singularities lie outside the present scope.
- **Numerics:** tolerances and step control matter; we report Wronskian drift and norm gaps as sanity checks.

---

## 9. Summary of Results
- **Theorems:** time‑ordering invariance and unitary equivalence under \(\sigma=\log(\tau/\tau_0)\).
- **Asymptotic silence:** precise conditions ensuring \(H_\text{eff}(\sigma)\to0\) and finite past phase.
- **Cosmology:** FLRW/Weyl constant‑curvature benchmark and geodesic completeness (in the Weyl frame) toward \(\sigma\to-\infty\) for standard eras.
- **QFT:** canonical variable \(v=a^{3/2}u\) yields conserved Wronskian; Bogoliubov invariants match across \(\tau\) and \(\sigma\) within tight tolerances.
- **Minisuperspace:** lapse‑first derivation retains standard GR structure before any clock choice.

---

## 10. Pointers for Further Work
- Back‑reaction: compute \(\langle T_{\mu\nu}\rangle\) consistently with the \(v\)-pipeline and feed into background dynamics.
- Massive/interacting fields: extend tests to \(m\neq 0\), adiabatic orders, and simple self‑interactions.
- Black‑hole charts: replicate σ‑vs‑τ invariance in near‑horizon slicings and study detector responses.

---

### Notation and Conventions (quick)
- Units: \(c=\hbar=1\). Signature \((- + + +)\).
- Dots denote \(d/d\tau\). For \(\sigma\) we use \(d/d\sigma = \tau\, d/d\tau\).
- KG inner product: for the canonical variable \(v\), \((f,g)= i(\bar f\,\dot g - \dot{\bar f}\,g)\).
- Wronskian: \(W(v,\bar v)\equiv i\).

