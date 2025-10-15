#!/usr/bin/env python3

import numpy as np

def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + 0.5*h, y + 0.5*h*k1)
    k3 = f(x + 0.5*h, y + 0.5*h*k2)
    k4 = f(x + h,     y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def test_qft_fixed():
    print("Testing improved QFT mode comparison...")
    
    tau0 = 1.0
    p = 1.5
    k = 1.0
    t_i, t_f = 0.1, 1.0
    N = 500
    
    # Reference: linear t-grid
    t_grid = np.linspace(t_i, t_f, N+1)
    
    # Initial conditions
    u0, v0 = 1.0, 0.0
    
    # τ-system: u'' + (3p/t)u' + (k²/t^(2p))u = 0
    def f_tau(t, y):
        u, v = y[0], y[1]
        du = v
        dv = -(3*p/t)*v - (k**2 / t**(2*p))*u
        return np.array([du, dv])
    
    # Solve τ-system
    y_tau = np.array([u0, v0])
    U_tau = np.zeros_like(t_grid)
    U_tau[0] = u0
    
    dt = (t_f - t_i) / N
    for i in range(N):
        y_tau = rk4_step(f_tau, t_grid[i], y_tau, dt)
        U_tau[i+1] = y_tau[0]
    
    # σ-system with matched grid
    s_grid = np.log(t_grid / tau0)
    
    def f_sigma(s, Y):
        t = tau0 * np.exp(s)
        u, w = Y[0], Y[1]
        
        # USER'S EXACT DERIVATION: u'' + (1-3p)u' + t²Ω²(t)u = 0
        # Where Ω²(t) = k²/t^(2p) for FLRW
        Omega2_t = (k**2) / (t**(2*p))
            
        du_ds = w
        dw_ds = -(1 - 3*p)*w - (t**2) * Omega2_t * u
        return np.array([du_ds, dw_ds])
    
    # Solve σ-system with matched times
    Y_sig = np.array([u0, t_i * v0])  # Corrected IC
    U_sig = np.zeros_like(s_grid)
    U_sig[0] = u0
    
    for i in range(N):
        ds = s_grid[i+1] - s_grid[i]
        Y_sig = rk4_step(f_sigma, s_grid[i], Y_sig, ds)
        U_sig[i+1] = Y_sig[0]
    
    # Compare at matched times
    diff = np.max(np.abs(np.abs(U_tau) - np.abs(U_sig)))
    print(f"Max difference: {diff:.8e}")
    
    if diff < 1e-4:
        print("EXCELLENT: High precision agreement")
    elif diff < 1e-3:
        print("GOOD: Acceptable numerical precision")
    else:
        print(f"ISSUES: Large difference {diff:.6e}")

if __name__ == "__main__":
    test_qft_fixed()