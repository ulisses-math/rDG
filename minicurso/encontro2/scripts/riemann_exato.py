# -*- coding: utf-8 -*-
"""
Solver de Riemann exato para as equações de Euler 1D (gás ideal), vetorizado.
Segue Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics", cap. 4.

Uso básico:
    from riemann_exato import riemann_exato, amostrar
    ps, us = riemann_exato(rhoL, uL, pL, rhoR, uR, pR)      # estado estrela
    rho, u, p = amostrar(S, rhoL, uL, pL, rhoR, uR, pR)      # solução em S = x/t

Todos os argumentos podem ser escalares ou arrays numpy do mesmo formato
(um problema de Riemann por face, por exemplo).
"""
import numpy as np

GAMMA = 1.4


def som(rho, p, gamma=GAMMA):
    return np.sqrt(gamma * p / rho)


def _fK(p, rhoK, pK, aK, gamma):
    """Função f_K(p) (Toro eq. 4.6-4.7) e sua derivada."""
    g = gamma
    AK = 2.0 / ((g + 1.0) * rhoK)
    BK = (g - 1.0) / (g + 1.0) * pK
    choque = p > pK
    # ramo de choque
    raiz = np.sqrt(AK / (p + BK))
    f_ch = (p - pK) * raiz
    df_ch = raiz * (1.0 - 0.5 * (p - pK) / (BK + p))
    # ramo de rarefação
    pr = p / pK
    f_ra = 2.0 * aK / (g - 1.0) * (pr ** ((g - 1.0) / (2.0 * g)) - 1.0)
    df_ra = 1.0 / (rhoK * aK) * pr ** (-(g + 1.0) / (2.0 * g))
    return np.where(choque, f_ch, f_ra), np.where(choque, df_ch, df_ra)


def riemann_exato(rhoL, uL, pL, rhoR, uR, pR, gamma=GAMMA, tol=1e-10, itmax=60):
    """Pressão e velocidade do estado estrela (Newton-Raphson em f(p) = 0)."""
    rhoL, uL, pL, rhoR, uR, pR = np.broadcast_arrays(*[np.asarray(a, dtype=float)
                                                      for a in (rhoL, uL, pL, rhoR, uR, pR)])
    aL, aR = som(rhoL, pL, gamma), som(rhoR, pR, gamma)
    du = uR - uL
    if np.any(2.0 * (aL + aR) / (gamma - 1.0) <= du):
        raise ValueError("Dado inicial gera vácuo: a condição de positividade da pressão falha.")
    # estimativa inicial: solver de Riemann linearizado em primitivas (PVRS), com piso
    p0 = 0.5 * (pL + pR) - 0.125 * du * (rhoL + rhoR) * (aL + aR)
    p0 = np.maximum(p0, 1e-8 * np.minimum(pL, pR))
    p = p0.copy()
    for _ in range(itmax):
        fL, dfL = _fK(p, rhoL, pL, aL, gamma)
        fR, dfR = _fK(p, rhoR, pR, aR, gamma)
        f = fL + fR + du
        dp = f / (dfL + dfR)
        pn = np.maximum(p - dp, 1e-12 * np.minimum(pL, pR))
        conv = 2.0 * np.abs(pn - p) / (pn + p)
        p = pn
        if np.all(conv < tol):
            break
    fL, _ = _fK(p, rhoL, pL, aL, gamma)
    fR, _ = _fK(p, rhoR, pR, aR, gamma)
    u = 0.5 * (uL + uR) + 0.5 * (fR - fL)
    return p, u


def amostrar(S, rhoL, uL, pL, rhoR, uR, pR, gamma=GAMMA, ps=None, us=None):
    """Solução exata (rho,u,p) na razão S = x/t (Toro, sec. 4.5)."""
    S, rhoL, uL, pL, rhoR, uR, pR = np.broadcast_arrays(*[np.asarray(a, dtype=float)
                                                         for a in (S, rhoL, uL, pL, rhoR, uR, pR)])
    g = gamma
    if ps is None:
        ps, us = riemann_exato(rhoL, uL, pL, rhoR, uR, pR, gamma)
    aL, aR = som(rhoL, pL, g), som(rhoR, pR, g)
    g1 = (g - 1.0) / (g + 1.0)
    rho = np.empty_like(S); u = np.empty_like(S); p = np.empty_like(S)

    # ---- lado esquerdo (S <= u*) ----
    esq = S <= us
    # choque à esquerda
    SL = uL - aL * np.sqrt((g + 1.0) / (2.0 * g) * ps / pL + (g - 1.0) / (2.0 * g))
    rhoLs_ch = rhoL * (ps / pL + g1) / (g1 * ps / pL + 1.0)
    # rarefação à esquerda
    SHL = uL - aL
    aLs = aL * (ps / pL) ** ((g - 1.0) / (2.0 * g))
    STL = us - aLs
    rhoLs_ra = rhoL * (ps / pL) ** (1.0 / g)
    fator = 2.0 / (g + 1.0) + (g - 1.0) / ((g + 1.0) * aL) * (uL - S)
    fator = np.maximum(fator, 1e-300)
    rho_fanL = rhoL * fator ** (2.0 / (g - 1.0))
    u_fanL = 2.0 / (g + 1.0) * (aL + 0.5 * (g - 1.0) * uL + S)
    p_fanL = pL * fator ** (2.0 * g / (g - 1.0))

    choqueL = ps > pL
    # choque
    m = esq & choqueL & (S <= SL)
    rho[m], u[m], p[m] = rhoL[m], uL[m], pL[m]
    m = esq & choqueL & (S > SL)
    rho[m], u[m], p[m] = rhoLs_ch[m], us[m], ps[m]
    # rarefação
    m = esq & ~choqueL & (S <= SHL)
    rho[m], u[m], p[m] = rhoL[m], uL[m], pL[m]
    m = esq & ~choqueL & (S > SHL) & (S < STL)
    rho[m], u[m], p[m] = rho_fanL[m], u_fanL[m], p_fanL[m]
    m = esq & ~choqueL & (S >= STL)
    rho[m], u[m], p[m] = rhoLs_ra[m], us[m], ps[m]

    # ---- lado direito (S > u*) ----
    dir_ = ~esq
    SR = uR + aR * np.sqrt((g + 1.0) / (2.0 * g) * ps / pR + (g - 1.0) / (2.0 * g))
    rhoRs_ch = rhoR * (ps / pR + g1) / (g1 * ps / pR + 1.0)
    SHR = uR + aR
    aRs = aR * (ps / pR) ** ((g - 1.0) / (2.0 * g))
    STR = us + aRs
    rhoRs_ra = rhoR * (ps / pR) ** (1.0 / g)
    fator = 2.0 / (g + 1.0) - (g - 1.0) / ((g + 1.0) * aR) * (uR - S)
    fator = np.maximum(fator, 1e-300)
    rho_fanR = rhoR * fator ** (2.0 / (g - 1.0))
    u_fanR = 2.0 / (g + 1.0) * (-aR + 0.5 * (g - 1.0) * uR + S)
    p_fanR = pR * fator ** (2.0 * g / (g - 1.0))

    choqueR = ps > pR
    m = dir_ & choqueR & (S >= SR)
    rho[m], u[m], p[m] = rhoR[m], uR[m], pR[m]
    m = dir_ & choqueR & (S < SR)
    rho[m], u[m], p[m] = rhoRs_ch[m], us[m], ps[m]
    m = dir_ & ~choqueR & (S >= SHR)
    rho[m], u[m], p[m] = rhoR[m], uR[m], pR[m]
    m = dir_ & ~choqueR & (S > STR) & (S < SHR)
    rho[m], u[m], p[m] = rho_fanR[m], u_fanR[m], p_fanR[m]
    m = dir_ & ~choqueR & (S <= STR)
    rho[m], u[m], p[m] = rhoRs_ra[m], us[m], ps[m]
    return rho, u, p


def velocidades_ondas(rhoL, uL, pL, rhoR, uR, pR, gamma=GAMMA):
    """Velocidades das ondas (cabeça/cauda das rarefações ou choques) e do contato."""
    g = gamma
    ps, us = riemann_exato(rhoL, uL, pL, rhoR, uR, pR, g)
    aL, aR = som(rhoL, pL, g), som(rhoR, pR, g)
    if ps > pL:
        esq = (uL - aL * np.sqrt((g + 1) / (2 * g) * ps / pL + (g - 1) / (2 * g)),) * 2
    else:
        esq = (uL - aL, us - aL * (ps / pL) ** ((g - 1) / (2 * g)))
    if ps > pR:
        dire = (uR + aR * np.sqrt((g + 1) / (2 * g) * ps / pR + (g - 1) / (2 * g)),) * 2
    else:
        dire = (us + aR * (ps / pR) ** ((g - 1) / (2 * g)), uR + aR)
    return dict(ps=float(ps), us=float(us), esq=tuple(map(float, esq)),
                contato=float(us), dir=tuple(map(float, dire)))


if __name__ == "__main__":
    # teste: tubo de choque de Sod
    info = velocidades_ondas(1.0, 0.0, 1.0, 0.125, 0.0, 0.1)
    print("Sod:", info)
    rho, u, p = amostrar(np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]), 1.0, 0.0, 1.0, 0.125, 0.0, 0.1)
    print("rho:", rho)
    print("u  :", u)
    print("p  :", p)
    # teste: problema 123
    print("123:", velocidades_ondas(1.0, -2.0, 0.4, 1.0, 2.0, 0.4))
