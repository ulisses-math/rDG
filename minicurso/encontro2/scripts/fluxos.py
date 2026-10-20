# -*- coding: utf-8 -*-
"""
Fluxos numéricos de volumes finitos (primeira ordem, Euler explícito no tempo)
para a equação de Burgers e para as equações de Euler 1D (gás ideal).

Burgers:  godunov_b, roe_b (com ou sem correção de entropia), rusanov_b, lf_b
Euler:    godunov_e (solver exato), roe_e (Roe + Harten-Hyman), hll_e, hllc_e, rusanov_e, lf_e

Todos os fluxos recebem os estados à esquerda e à direita de cada face (arrays)
e devolvem o fluxo numérico F(uL, uR) em cada face.
"""
import numpy as np
from riemann_exato import riemann_exato, amostrar, som, GAMMA

# ============================================================== Burgers
def f_b(u):
    return 0.5 * u * u


def godunov_b(uL, uR, **kw):
    """Fluxo de Godunov para f(u) = u^2/2 (fórmula do máximo/mínimo)."""
    return np.maximum(f_b(np.maximum(uL, 0.0)), f_b(np.minimum(uR, 0.0)))


def roe_b(uL, uR, fix=None, **kw):
    """Fluxo de Roe: a = (uL+uR)/2.  fix=None | 'hh' (Harten-Hyman) | 'harten'."""
    a = 0.5 * (uL + uR)
    du = uR - uL
    absa = np.abs(a)
    if fix == 'hh':
        # onda transônica: lamL = uL < 0 < uR = lamR; reparte o salto em duas ondas
        trans = (uL < 0.0) & (uR > 0.0)
        beta = np.where(trans, (uR - a) / np.where(trans, uR - uL, 1.0), 0.0)
        lam_menos = np.where(trans, beta * uL, np.minimum(a, 0.0))
        return f_b(uL) + lam_menos * du
    if fix == 'harten':
        delta = np.maximum(0.0, np.maximum(a - uL, uR - a))
        psi = np.where(absa < delta, (a * a + delta * delta) / (2.0 * np.where(delta > 0, delta, 1.0)), absa)
        return 0.5 * (f_b(uL) + f_b(uR)) - 0.5 * psi * du
    return 0.5 * (f_b(uL) + f_b(uR)) - 0.5 * absa * du


def rusanov_b(uL, uR, **kw):
    s = np.maximum(np.abs(uL), np.abs(uR))
    return 0.5 * (f_b(uL) + f_b(uR)) - 0.5 * s * (uR - uL)


def lf_b(uL, uR, h=None, dt=None, **kw):
    return 0.5 * (f_b(uL) + f_b(uR)) - 0.5 * (h / dt) * (uR - uL)


def viscoso_b(uL, uR, Q=None, **kw):
    """Forma 'central + dissipação' com coeficiente Q dado (número ou função)."""
    q = Q(uL, uR) if callable(Q) else Q
    return 0.5 * (f_b(uL) + f_b(uR)) - 0.5 * q * (uR - uL)


def exato_riemann_b(x, t, uL, uR):
    """Solução exata do problema de Riemann de Burgers (salto em x=0)."""
    x = np.asarray(x, dtype=float)
    if uL > uR:                       # choque
        s = 0.5 * (uL + uR)
        return np.where(x < s * t, uL, uR)
    if t == 0:
        return np.where(x < 0, uL, uR)
    return np.clip(x / t, uL, uR)     # rarefação (inclui o caso transônico)


def avancar_b(u0, x, t_fim, fluxo, cfl=0.8, bc='transmissivo', **kw):
    """Integra Burgers com volumes finitos de primeira ordem. u0: médias iniciais."""
    u = np.array(u0, dtype=float)
    h = x[1] - x[0]
    t = 0.0
    passos = 0
    while t < t_fim - 1e-14:
        smax = max(np.max(np.abs(u)), 1e-12)
        dt = min(cfl * h / smax, t_fim - t)
        if bc == 'periodico':
            ue = np.concatenate(([u[-1]], u, [u[0]]))
        else:
            ue = np.concatenate(([u[0]], u, [u[-1]]))
        F = fluxo(ue[:-1], ue[1:], h=h, dt=dt, **kw)
        u = u - dt / h * (F[1:] - F[:-1])
        t += dt
        passos += 1
    return u, passos


# ============================================================== Euler
def prim2cons(rho, u, p, gamma=GAMMA):
    E = p / (gamma - 1.0) + 0.5 * rho * u * u
    return np.array([rho, rho * u, E])


def cons2prim(U, gamma=GAMMA):
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1.0) * (U[2] - 0.5 * rho * u * u)
    return rho, u, p


def f_e(U, gamma=GAMMA):
    rho, u, p = cons2prim(U, gamma)
    return np.array([rho * u, rho * u * u + p, u * (U[2] + p)])


def godunov_e(UL, UR, gamma=GAMMA, **kw):
    """Godunov com o solver de Riemann exato: F = f(u_RP(0))."""
    rL, uL, pL = cons2prim(UL, gamma)
    rR, uR, pR = cons2prim(UR, gamma)
    S = np.zeros_like(rL)
    r, u, p = amostrar(S, rL, uL, pL, rR, uR, pR, gamma)
    return f_e(prim2cons(r, u, p, gamma), gamma)


def medias_roe(UL, UR, gamma=GAMMA):
    rL, uL, pL = cons2prim(UL, gamma)
    rR, uR, pR = cons2prim(UR, gamma)
    HL = (UL[2] + pL) / rL
    HR = (UR[2] + pR) / rR
    sL, sR = np.sqrt(rL), np.sqrt(rR)
    ut = (sL * uL + sR * uR) / (sL + sR)
    Ht = (sL * HL + sR * HR) / (sL + sR)
    at = np.sqrt((gamma - 1.0) * (Ht - 0.5 * ut * ut))
    return ut, Ht, at


def ondas_roe(UL, UR, gamma=GAMMA):
    """Autovalores, autovetores e intensidades das ondas do problema linearizado (Toro 11.2)."""
    ut, Ht, at = medias_roe(UL, UR, gamma)
    dU = UR - UL
    lam = np.array([ut - at, ut, ut + at])
    K = np.array([[np.ones_like(ut), ut - at, Ht - ut * at],
                  [np.ones_like(ut), ut, 0.5 * ut * ut],
                  [np.ones_like(ut), ut + at, Ht + ut * at]])          # K[p] = autovetor p
    a2 = (gamma - 1.0) / at**2 * (dU[0] * (Ht - ut * ut) + ut * dU[1] - dU[2])
    a1 = 1.0 / (2.0 * at) * (dU[0] * (ut + at) - dU[1] - at * a2)
    a3 = dU[0] - (a1 + a2)
    alpha = np.array([a1, a2, a3])
    return lam, K, alpha


def roe_e(UL, UR, gamma=GAMMA, fix='hh', **kw):
    """Fluxo de Roe para Euler 1D, com correção de entropia de Harten-Hyman (fix='hh'),
    de Harten (fix='harten', delta = abertura do leque) ou sem correção (fix=None)."""
    lam, K, alpha = ondas_roe(UL, UR, gamma)
    FL, FR = f_e(UL, gamma), f_e(UR, gamma)
    if fix is None:
        diss = np.zeros_like(FL)
        for p in range(3):
            diss += np.abs(lam[p]) * alpha[p] * K[p]
        return 0.5 * (FL + FR) - 0.5 * diss
    # estados intermediários do leque linearizado e velocidades características reais
    rL, uL, pL = cons2prim(UL, gamma); aL = som(rL, pL, gamma)
    rR, uR, pR = cons2prim(UR, gamma); aR = som(rR, pR, gamma)
    UsL = UL + alpha[0] * K[0]
    UsR = UR - alpha[2] * K[2]
    rsL, usL, psL = cons2prim(UsL, gamma)
    rsR, usR, psR = cons2prim(UsR, gamma)
    ok_L = (rsL > 0) & (psL > 0)
    ok_R = (rsR > 0) & (psR > 0)
    asL = np.sqrt(gamma * np.where(ok_L, psL, pL) / np.where(ok_L, rsL, rL))
    asR = np.sqrt(gamma * np.where(ok_R, psR, pR) / np.where(ok_R, rsR, rR))
    lam1L, lam1R = uL - aL, usL - asL         # onda 1: esquerda / direita da onda
    lam3L, lam3R = usR + asR, uR + aR         # onda 3
    lamL = np.array([lam1L, lam[1], lam3L])
    lamR = np.array([lam1R, lam[1], lam3R])
    if fix == 'harten':
        diss = np.zeros_like(FL)
        for p in range(3):
            delta = np.maximum(0.0, np.maximum(lam[p] - lamL[p], lamR[p] - lam[p]))
            absl = np.abs(lam[p])
            psi = np.where(absl < delta, (lam[p]**2 + delta**2) / (2.0 * np.where(delta > 0, delta, 1.0)), absl)
            diss += psi * alpha[p] * K[p]
        return 0.5 * (FL + FR) - 0.5 * diss
    # Harten-Hyman: forma de flutuações, F = FL + sum lam^- alpha K
    F = FL.copy()
    for p in range(3):
        trans = (lamL[p] < 0.0) & (lamR[p] > 0.0)
        den = np.where(trans, lamR[p] - lamL[p], 1.0)
        beta = (lamR[p] - lam[p]) / den
        lam_menos = np.where(trans, beta * lamL[p], np.minimum(lam[p], 0.0))
        F += lam_menos * alpha[p] * K[p]
    return F


def velocidades_hll(UL, UR, gamma=GAMMA):
    rL, uL, pL = cons2prim(UL, gamma); aL = som(rL, pL, gamma)
    rR, uR, pR = cons2prim(UR, gamma); aR = som(rR, pR, gamma)
    ut, Ht, at = medias_roe(UL, UR, gamma)
    SL = np.minimum(uL - aL, ut - at)
    SR = np.maximum(uR + aR, ut + at)
    return SL, SR


def hll_e(UL, UR, gamma=GAMMA, **kw):
    SL, SR = velocidades_hll(UL, UR, gamma)
    FL, FR = f_e(UL, gamma), f_e(UR, gamma)
    Fhll = (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)
    return np.where(SL >= 0, FL, np.where(SR <= 0, FR, Fhll))


def hllc_e(UL, UR, gamma=GAMMA, **kw):
    SL, SR = velocidades_hll(UL, UR, gamma)
    rL, uL, pL = cons2prim(UL, gamma)
    rR, uR, pR = cons2prim(UR, gamma)
    FL, FR = f_e(UL, gamma), f_e(UR, gamma)
    Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR))

    def estrela(U, r, u, p, SK):
        fac = r * (SK - u) / (SK - Ss)
        return np.array([fac, fac * Ss, fac * (U[2] / r + (Ss - u) * (Ss + p / (r * (SK - u))))])
    UsL = estrela(UL, rL, uL, pL, SL)
    UsR = estrela(UR, rR, uR, pR, SR)
    FsL = FL + SL * (UsL - UL)
    FsR = FR + SR * (UsR - UR)
    return np.where(SL >= 0, FL, np.where(Ss >= 0, FsL, np.where(SR > 0, FsR, FR)))


def rusanov_e(UL, UR, gamma=GAMMA, **kw):
    rL, uL, pL = cons2prim(UL, gamma)
    rR, uR, pR = cons2prim(UR, gamma)
    s = np.maximum(np.abs(uL) + som(rL, pL, gamma), np.abs(uR) + som(rR, pR, gamma))
    return 0.5 * (f_e(UL, gamma) + f_e(UR, gamma)) - 0.5 * s * (UR - UL)


def lf_e(UL, UR, gamma=GAMMA, h=None, dt=None, **kw):
    return 0.5 * (f_e(UL, gamma) + f_e(UR, gamma)) - 0.5 * (h / dt) * (UR - UL)


class EstadoNaoFisico(Exception):
    pass


def avancar_e(W0, x, t_fim, fluxo, cfl=0.8, gamma=GAMMA, bc='transmissivo', **kw):
    """Integra Euler 1D com volumes finitos de primeira ordem.
    W0 = (rho, u, p) arrays iniciais (médias por célula). Devolve (rho,u,p), passos."""
    U = prim2cons(*W0, gamma=gamma)
    h = x[1] - x[0]
    t = 0.0
    passos = 0
    while t < t_fim - 1e-14:
        rho, u, p = cons2prim(U, gamma)
        if np.any(rho <= 0) or np.any(p <= 0) or np.any(~np.isfinite(rho)):
            raise EstadoNaoFisico(f"densidade ou pressão não positiva em t={t:.4f}, passo {passos}")
        smax = np.max(np.abs(u) + som(rho, p, gamma))
        dt = min(cfl * h / smax, t_fim - t)
        if bc == 'periodico':
            Ue = np.concatenate((U[:, -1:], U, U[:, :1]), axis=1)
        else:
            Ue = np.concatenate((U[:, :1], U, U[:, -1:]), axis=1)
        F = fluxo(Ue[:, :-1], Ue[:, 1:], gamma=gamma, h=h, dt=dt, **kw)
        U = U - dt / h * (F[:, 1:] - F[:, :-1])
        t += dt
        passos += 1
    return cons2prim(U, gamma), passos


def malha(a, b, N):
    h = (b - a) / N
    return a + (np.arange(N) + 0.5) * h, h


def riemann_inicial(x, x0, WL, WR):
    rho = np.where(x < x0, WL[0], WR[0]).astype(float)
    u = np.where(x < x0, WL[1], WR[1]).astype(float)
    p = np.where(x < x0, WL[2], WR[2]).astype(float)
    return rho, u, p


SOD = ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1))
P123 = ((1.0, -2.0, 0.4), (1.0, 2.0, 0.4))


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    # --- propriedade (i) da matriz de Roe: sum lam alpha K = Delta f, sum alpha K = Delta U
    rL, uL, pL = rng.uniform(0.2, 2, 50), rng.uniform(-2, 2, 50), rng.uniform(0.1, 3, 50)
    rR, uR, pR = rng.uniform(0.2, 2, 50), rng.uniform(-2, 2, 50), rng.uniform(0.1, 3, 50)
    UL, UR = prim2cons(rL, uL, pL), prim2cons(rR, uR, pR)
    lam, K, alpha = ondas_roe(UL, UR)
    dF = sum(lam[p] * alpha[p] * K[p] for p in range(3))
    dU = sum(alpha[p] * K[p] for p in range(3))
    print("Roe (i): max|sum lam alpha K - dF| =", np.max(np.abs(dF - (f_e(UR) - f_e(UL)))))
    print("Roe    : max|sum alpha K - dU|     =", np.max(np.abs(dU - (UR - UL))))
    # consistência de todos os fluxos
    for nome, fl in [("godunov", godunov_e), ("roe", roe_e), ("hll", hll_e), ("hllc", hllc_e), ("rusanov", rusanov_e)]:
        err = np.max(np.abs(fl(UL, UL) - f_e(UL)))
        print(f"consistência {nome:8s}: max|F(u,u)-f(u)| = {err:.2e}")
    # --- Sod com cada fluxo: tempo e erro L1 em rho
    for N in (100, 400):
        x, h = malha(0.0, 1.0, N)
        W0 = riemann_inicial(x, 0.5, *SOD)
        rex, uex, pex = amostrar((x - 0.5) / 0.2, *SOD[0], *SOD[1])
        for nome, fl in [("godunov", godunov_e), ("roe", roe_e), ("hll", hll_e), ("hllc", hllc_e), ("rusanov", rusanov_e), ("lf", lf_e)]:
            t0 = time.time()
            (r, u, p), n = avancar_e(W0, x, 0.2, fl, cfl=0.8)
            E = h * np.sum(np.abs(r - rex))
            print(f"Sod N={N:4d} {nome:8s}: passos={n:4d}  E1(rho)={E:.4e}  tempo={time.time()-t0:.2f}s")
    # --- problema 123: quem sobrevive?
    for vel in (2.0, 3.0, 4.0, 5.0):
        WL, WR = (1.0, -vel, 0.4), (1.0, vel, 0.4)
        for N in (100, 400):
            x, h = malha(0.0, 1.0, N)
            W0 = riemann_inicial(x, 0.5, WL, WR)
            for nome, fl, kw in [("godunov", godunov_e, {}), ("roe-hh", roe_e, {}), ("roe-sem", roe_e, {'fix': None}),
                                 ("roe-harten", roe_e, {'fix': 'harten'}), ("hll", hll_e, {}), ("hllc", hllc_e, {}), ("rusanov", rusanov_e, {})]:
                try:
                    (r, u, p), n = avancar_e(W0, x, 0.15, fl, cfl=0.8, **kw)
                    res = f"ok (min rho={r.min():.3e}, min p={p.min():.3e})"
                except (EstadoNaoFisico, ValueError, FloatingPointError) as e:
                    res = f"FALHA: {e}"
                print(f"123 u=±{vel} N={N:4d} {nome:10s}: {res}")
