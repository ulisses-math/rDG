# -*- coding: utf-8 -*-
"""
rDG(P0P1) + HLLC + minmod + BDF2 para as equacoes de Euler 1D.
Implementacao de referencia do tutorial (todas as figuras saem daqui).
"""
import os, sys
import numpy as np
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import LinearOperator, gmres

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "encontro2", "scripts"))
from riemann_exato import amostrar  # solver de Riemann exato (referencia)

GAMMA = 1.4

# ----------------------------------------------------------------- variaveis
def conserved_to_primitive(U):
    rho = U[0]
    v = U[1] / rho
    p = (GAMMA - 1.0) * (U[2] - 0.5 * rho * v * v)
    return rho, v, p


def primitive_to_conserved(rho, v, p):
    return np.array([rho, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * v * v])


def sound_speed(rho, p):
    return np.sqrt(GAMMA * p / rho)


def euler_flux(U):
    rho, v, p = conserved_to_primitive(U)
    return np.array([rho * v, rho * v * v + p, (U[2] + p) * v])


# ----------------------------------------------------------- Passo 1: minmod
def minmod(a, b):
    """minmod(a, b): o de menor modulo se os sinais coincidem, zero caso contrario."""
    same_sign = (a * b > 0)
    return np.where(same_sign, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def ghost(U, bc):
    """Acrescenta uma celula fantasma de cada lado."""
    if bc == "periodic":
        return np.concatenate((U[:, -1:], U, U[:, :1]), axis=1)
    return np.concatenate((U[:, :1], U, U[:, -1:]), axis=1)        # transmissivo


def reconstruct(U, bc="transmissive", limiter="minmod"):
    """Coeficiente de inclinacao u_hat_i (valor de face = media +- u_hat)."""
    Ug = ghost(U, bc)
    delta_plus = Ug[:, 2:] - Ug[:, 1:-1]      # Delta_+ = u_{i+1} - u_i
    delta_minus = Ug[:, 1:-1] - Ug[:, :-2]    # Delta_- = u_i - u_{i-1}
    if limiter == "none":                     # diferenca central, sem limitador
        return 0.25 * (delta_plus + delta_minus)
    if limiter == "zero":                     # P0 puro
        return np.zeros_like(U)
    return 0.5 * minmod(delta_plus, delta_minus)


# --------------------------------------------------- Passo 2: estados de face
def interface_states(U, S, bc="transmissive"):
    """Estados u^- (esquerda) e u^+ (direita) em todas as faces, inclusive as de contorno."""
    Ug, Sg = ghost(U, bc), ghost(S, bc)
    if bc != "periodic":
        Sg[:, 0] = 0.0; Sg[:, -1] = 0.0
    uL = Ug[:, :-1] + Sg[:, :-1]              # u^-_{i+1/2} = u_i + u_hat_i
    uR = Ug[:, 1:] - Sg[:, 1:]                # u^+_{i+1/2} = u_{i+1} - u_hat_{i+1}
    return uL, uR


# ------------------------------------------------------------ Passo 3: HLLC
def hllc_flux(uL, uR):
    """Fluxo HLLC (Toro) vetorizado. Devolve o fluxo e o ramo usado (0:L, 1:*L, 2:*R, 3:R)."""
    rhoL, vL, pL = conserved_to_primitive(uL)
    rhoR, vR, pR = conserved_to_primitive(uR)
    aL, aR = sound_speed(rhoL, pL), sound_speed(rhoR, pR)
    fL, fR = euler_flux(uL), euler_flux(uR)
    # velocidades das ondas externas (Davis)
    SL = np.minimum(vL - aL, vR - aR)
    SR = np.maximum(vL + aL, vR + aR)
    # velocidade da onda de contato
    num = pR - pL + rhoL * vL * (SL - vL) - rhoR * vR * (SR - vR)
    den = rhoL * (SL - vL) - rhoR * (SR - vR)
    S_star = num / den
    # estados estrela
    def star(u, rho, v, p, S):
        fac = rho * (S - v) / (S - S_star)
        return np.array([fac, fac * S_star, fac * (u[2] / rho + (S_star - v) * (S_star + p / (rho * (S - v))))])
    u_starL = star(uL, rhoL, vL, pL, SL)
    u_starR = star(uR, rhoR, vR, pR, SR)
    f_starL = fL + SL * (u_starL - uL)
    f_starR = fR + SR * (u_starR - uR)
    F = np.where(SL >= 0, fL, np.where(S_star >= 0, f_starL, np.where(SR >= 0, f_starR, fR)))
    regime = np.where(SL >= 0, 0, np.where(S_star >= 0, 1, np.where(SR >= 0, 2, 3)))
    return F, regime, (SL, S_star, SR)


# ------------------------------------------------------- operador espacial L
def compute_rhs(U, dx, bc="transmissive", limiter="minmod"):
    S = reconstruct(U, bc, limiter)
    uL, uR = interface_states(U, S, bc)
    F, _, _ = hllc_flux(uL, uR)
    return -(F[:, 1:] - F[:, :-1]) / dx


# ---------------------------------------------------- Passo 4: BDF1 e BDF2
def residual(U, U_n, U_nm1, dx, dt, bc, limiter):
    L = compute_rhs(U, dx, bc, limiter)
    if U_nm1 is None:                                   # BDF1 (Euler implicito)
        return U - U_n - dt * L
    return 3.0 * U - 4.0 * U_n + U_nm1 - 2.0 * dt * L     # BDF2


def step_fixed_point(U_n, U_nm1, dx, dt, bc="transmissive", limiter="minmod", tol=1e-10, max_iter=50):
    """Iteracao de ponto fixo (dU = -F/3 ou -F/1), como no listing original."""
    c = 1.0 if U_nm1 is None else 3.0
    U = U_n.copy(); hist = []
    for k in range(max_iter):
        F = residual(U, U_n, U_nm1, dx, dt, bc, limiter)
        hist.append(np.max(np.abs(F)))
        if hist[-1] < tol or not np.isfinite(hist[-1]):
            break
        U = U - F / c
    return U, hist


def step_newton_krylov(U_n, U_nm1, dx, dt, bc="transmissive", limiter="minmod", tol=1e-10, max_iter=50):
    """Newton-Krylov sem Jacobiana (JFNK): GMRES com produto Jacobiana-vetor por diferenca finita."""
    shape = U_n.shape
    hist = []

    def Fres(x):
        r = residual(x.reshape(shape), U_n, U_nm1, dx, dt, bc, limiter).ravel()
        return r

    x = U_n.ravel().copy()
    for k in range(max_iter):
        F = Fres(x)
        hist.append(np.max(np.abs(F)))
        if hist[-1] < tol:
            break
        eps = 1e-7 * max(1.0, np.linalg.norm(x)) / max(1.0, 1.0)
        def Jv(v):
            nv = np.linalg.norm(v)
            if nv == 0:
                return np.zeros_like(v)
            e = eps / nv
            return (Fres(x + e * v) - F) / e
        J = LinearOperator((x.size, x.size), matvec=Jv, dtype=float)
        dx_, info = gmres(J, -F, rtol=1e-3, restart=60, maxiter=200)
        x = x + dx_
    return x.reshape(shape), hist


def time_advance(U0, dx, t_final, CFL=5.0, bc="transmissive", limiter="minmod", solver="jfnk", dt=None):
    """Avanco temporal com passo constante: BDF1 no primeiro passo, BDF2 depois."""
    U = U0.copy()
    if dt is None:
        rho, v, p = conserved_to_primitive(U)
        dt = CFL * dx / np.max(np.abs(v) + sound_speed(rho, p))
    nsteps = int(np.ceil(t_final / dt - 1e-12))
    dt = t_final / nsteps                                  # ajusta para terminar exatamente em t_final
    step = step_newton_krylov if solver == "jfnk" else step_fixed_point
    U_nm1 = None
    iters = []
    for n in range(nsteps):
        U_new, hist = step(U, U_nm1, dx, dt, bc, limiter)
        iters.append(len(hist))
        U_nm1, U = U, U_new
    return U, dt, iters


# -------------------------------------------------------------- problemas
def mesh(N, a=0.0, b=1.0):
    dx = (b - a) / N
    return a + (np.arange(N) + 0.5) * dx, dx


SOD_L, SOD_R = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)


def sod_initial(x):
    rho = np.where(x < 0.5, SOD_L[0], SOD_R[0]).astype(float)
    v = np.zeros_like(x)
    p = np.where(x < 0.5, SOD_L[2], SOD_R[2]).astype(float)
    return primitive_to_conserved(rho, v, p)


def sod_exact(x, t):
    rho, v, p = amostrar((x - 0.5) / t, *SOD_L, *SOD_R)
    return rho, v, p


def sod_exact_average(x, dx, t, nq=20):
    """Media exata por celula (quadratura fina) para o erro L1."""
    xi = (np.arange(nq) + 0.5) / nq - 0.5
    rho = np.zeros_like(x)
    for s in xi:
        r, _, _ = sod_exact(x + s * dx, t)
        rho += r / nq
    return rho


def density_wave_initial(x, amp=0.2):
    rho = 1.0 + amp * np.sin(2 * np.pi * x)
    return primitive_to_conserved(rho, np.ones_like(x), np.ones_like(x))


def density_wave_exact_average(x, dx, t, amp=0.2):
    """Media exata de rho = 1 + amp sin(2 pi (x - t)) em cada celula."""
    xl, xr = x - 0.5 * dx - t, x + 0.5 * dx - t
    return 1.0 + amp * (np.cos(2 * np.pi * xl) - np.cos(2 * np.pi * xr)) / (2 * np.pi * dx)


if __name__ == "__main__":
    # 1) ponto fixo vs JFNK no primeiro passo BDF2 com CFL = 5
    x, dx = mesh(100)
    U0 = sod_initial(x)
    rho, v, p = conserved_to_primitive(U0)
    dt = 5.0 * dx / np.max(np.abs(v) + sound_speed(rho, p))
    U1, h1 = step_newton_krylov(U0, None, dx, dt)
    _, hfp = step_fixed_point(U1, U0, dx, dt)
    _, hnk = step_newton_krylov(U1, U0, dx, dt)
    print("ponto fixo, CFL=5 (passo BDF2):", ["%.2e" % r for r in hfp[:8]])
    print("JFNK,       CFL=5 (passo BDF2):", ["%.2e" % r for r in hnk])
    for cfl in (0.5, 0.74, 0.76, 1.0):
        dtc = cfl * dx / np.max(np.abs(v) + sound_speed(rho, p))
        _, h = step_fixed_point(U1, U0, dx, dtc, max_iter=60)
        print(f"ponto fixo CFL={cfl}: {len(h)} iteracoes, residuo final {h[-1]:.2e}")
    # 2) Sod: erro L1 em rho para P0 e P0P1
    print("--- Sod, t=0.2, CFL=5 (BDF2 + JFNK) ---")
    for N in (50, 100, 200, 400, 800):
        x, dx = mesh(N)
        rex = sod_exact_average(x, dx, 0.2)
        for lim in ("zero", "minmod"):
            U, dt, it = time_advance(sod_initial(x), dx, 0.2, CFL=5.0, limiter=lim)
            r = conserved_to_primitive(U)[0]
            print(f"N={N:4d} {lim:7s}: E1={dx*np.sum(np.abs(r-rex)):.4e}  passos={len(it)}  Newton medio={np.mean(it):.1f}")
    # 3) onda de densidade suave (periodica): ordem
    print("--- onda de densidade, t=1, CFL=1 ---")
    for N in (20, 40, 80, 160, 320):
        x, dx = mesh(N)
        rex = density_wave_exact_average(x, dx, 1.0)
        for lim in ("zero", "minmod", "none"):
            U, dt, it = time_advance(density_wave_initial(x), dx, 1.0, CFL=1.0, bc="periodic", limiter=lim)
            r = conserved_to_primitive(U)[0]
            print(f"N={N:4d} {lim:7s}: E1={dx*np.sum(np.abs(r-rex)):.4e}")
