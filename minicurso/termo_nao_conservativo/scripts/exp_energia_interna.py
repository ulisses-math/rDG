# -*- coding: utf-8 -*-
"""
Experimento: Euler 1D escrito com a ENERGIA INTERNA (forma não conservativa),
    rho_t + (rho u)_x = 0
    (rho u)_t + (rho u^2 + p)_x = 0
    (rho e)_t + (rho e u)_x + p u_x = 0      <- produto não conservativo p u_x
comparado com a forma conservativa (energia total). Esquema de Rusanov de 1a ordem;
o produto p u_x é tratado de forma path-conservative com dois caminhos lineares:
    (I)  linear em (rho, u, p):          int p du = (pL+pR)/2 (uR-uL)
    (II) linear em (rho, rho u, rho e):  int p du por quadratura de Gauss
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "encontro2", "scripts"))
from riemann_exato import amostrar

G = 1.4
xg, wg = np.polynomial.legendre.leggauss(6)
sg, wg = 0.5 * (xg + 1.0), 0.5 * wg


def prim_nc(W):
    rho, m, Ei = W
    u = m / rho
    p = (G - 1.0) * Ei
    return rho, u, p


def fluxo_nc(W):
    rho, u, p = prim_nc(W)
    return np.array([rho * u, rho * u * u + p, W[2] * u])


def integral_pdu(WL, WR, caminho):
    rL, uL, pL = prim_nc(WL)
    rR, uR, pR = prim_nc(WR)
    if caminho == "I":
        return 0.5 * (pL + pR) * (uR - uL)
    dW = WR - WL
    tot = 0.0
    for s, w in zip(sg, wg):
        Ws = WL + s * dW
        rho, m, Ei = Ws
        p = (G - 1.0) * Ei
        dudS = (dW[1] * rho - m * dW[0]) / rho**2
        tot = tot + w * p * dudS
    return tot


def roda_nc(x, h, W0prim, t_fim, caminho, cfl=0.8):
    rho, u, p = [np.array(a, float) for a in W0prim]
    W = np.array([rho, rho * u, p / (G - 1.0)])
    t = 0.0
    while t < t_fim - 1e-14:
        rho, u, p = prim_nc(W)
        lam = np.abs(u) + np.sqrt(G * p / rho)
        dt = min(cfl * h / lam.max(), t_fim - t)
        We = np.concatenate((W[:, :1], W, W[:, -1:]), axis=1)
        WL, WR = We[:, :-1], We[:, 1:]
        lf = np.maximum(lam[np.r_[0, :len(lam)]], lam[np.r_[:len(lam), -1]])
        F = 0.5 * (fluxo_nc(WL) + fluxo_nc(WR)) - 0.5 * lf * (WR - WL)
        J = integral_pdu(WL, WR, caminho)          # salto não conservativo em cada face
        Dm = np.zeros_like(F); Dp = np.zeros_like(F)
        Dm[2] = 0.5 * J; Dp[2] = 0.5 * J           # repartição central (Rusanov)
        W = W - dt / h * (F[:, 1:] - F[:, :-1] + Dm[:, 1:] + Dp[:, :-1])
        t += dt
    return prim_nc(W)


def roda_cons(x, h, W0prim, t_fim, cfl=0.8):
    rho, u, p = [np.array(a, float) for a in W0prim]
    U = np.array([rho, rho * u, p / (G - 1.0) + 0.5 * rho * u * u])
    t = 0.0
    while t < t_fim - 1e-14:
        rho = U[0]; u = U[1] / rho; p = (G - 1.0) * (U[2] - 0.5 * rho * u * u)
        lam = np.abs(u) + np.sqrt(G * p / rho)
        dt = min(cfl * h / lam.max(), t_fim - t)
        Ue = np.concatenate((U[:, :1], U, U[:, -1:]), axis=1)
        UL, UR = Ue[:, :-1], Ue[:, 1:]
        def fl(V):
            r = V[0]; uu = V[1] / r; pp = (G - 1.0) * (V[2] - 0.5 * r * uu * uu)
            return np.array([r * uu, r * uu * uu + pp, uu * (V[2] + pp)])
        lf = np.maximum(lam[np.r_[0, :len(lam)]], lam[np.r_[:len(lam), -1]])
        F = 0.5 * (fl(UL) + fl(UR)) - 0.5 * lf * (UR - UL)
        U = U - dt / h * (F[:, 1:] - F[:, :-1])
        t += dt
    rho = U[0]; u = U[1] / rho; p = (G - 1.0) * (U[2] - 0.5 * rho * u * u)
    return rho, u, p


def caso(nome):
    if nome == "sod":
        return (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.2
    if nome == "forte":   # choque forte (razão de pressões 1000)
        return (1.0, 0.0, 1000.0), (1.0, 0.0, 1.0), 0.012


if __name__ == "__main__":
    for nome in ("sod", "forte"):
        WL, WR, tf = caso(nome)
        print(f"===== {nome} =====")
        for N in (400, 1600, 6400):
            h = 1.0 / N
            x = (np.arange(N) + 0.5) * h
            W0 = [np.where(x < 0.5, WL[k], WR[k]) for k in range(3)]
            rex, uex, pex = amostrar((x - 0.5) / tf, *WL, *WR)
            rc, uc, pc = roda_cons(x, h, W0, tf)
            r1, u1, p1 = roda_nc(x, h, W0, tf, "I")
            r2, u2, p2 = roda_nc(x, h, W0, tf, "II")
            E = lambda a, b: h * np.sum(np.abs(a - b))
            # posição do choque: último ponto onde rho > (rho*_R + rho_R)/2
            print(f"N={N:5d}  E1(rho): cons={E(rc,rex):.3e}  NC-I={E(r1,rex):.3e}  NC-II={E(r2,rex):.3e}"
                  f"   | NC-I vs NC-II = {E(r1,r2):.3e}")
            np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"res_{nome}_{N}.npz"),
                     x=x, rex=rex, pex=pex, uex=uex, rc=rc, pc=pc, r1=r1, p1=p1, r2=r2, p2=p2, u1=u1, u2=u2, uc=uc)
