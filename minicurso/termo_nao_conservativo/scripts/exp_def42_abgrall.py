# -*- coding: utf-8 -*-
"""
Teste da Definição 4.2 das notas (esquema P0P1 path-conservative) no modelo BN de 7 equações:
critério de Abgrall — com p e u uniformes nas duas fases e alpha variável, p e u devem
permanecer uniformes. Três variantes do mesmo esquema:
    A: Definição 4.2 (F, D+-, G_i com valores reconstruídos)
    B: sem o termo de volume G_i
    C: D+- calculados com as médias das células (G_i mantido) -> contagem dupla
Fluxo de Rusanov + flutuações path-conservative pelo caminho linear nas variáveis conservativas.
Fechamento BN (u_I, p_I) = (u_2, p_1); gas rígido (stiffened gas) nas duas fases.
"""
import numpy as np

g1, pinf1 = 4.4, 6.0e8      # fase 1: líquido
g2, pinf2 = 1.4, 0.0        # fase 2: gás
xq, wq = np.polynomial.legendre.leggauss(3); sq, wq = 0.5 * (xq + 1), 0.5 * wq     # caminho (faces)
sG = np.array([0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)]); wG = np.array([0.5, 0.5])  # G_i


def prim(U):
    a1 = U[0]; a2 = 1.0 - a1
    r1 = U[1] / a1; u1 = U[2] / U[1]; e1 = U[3] / U[1] - 0.5 * u1**2
    r2 = U[4] / a2; u2 = U[5] / U[4]; e2 = U[6] / U[4] - 0.5 * u2**2
    p1 = (g1 - 1) * r1 * e1 - g1 * pinf1
    p2 = (g2 - 1) * r2 * e2 - g2 * pinf2
    return a1, r1, u1, p1, a2, r2, u2, p2


def cons(a1, r1, u1, p1, r2, u2, p2):
    a2 = 1.0 - a1
    E1 = (p1 + g1 * pinf1) / ((g1 - 1) * r1) + 0.5 * u1**2
    E2 = (p2 + g2 * pinf2) / ((g2 - 1) * r2) + 0.5 * u2**2
    return np.array([a1, a1 * r1, a1 * r1 * u1, a1 * r1 * E1, a2 * r2, a2 * r2 * u2, a2 * r2 * E2])


def fluxo(U):
    a1, r1, u1, p1, a2, r2, u2, p2 = prim(U)
    return np.array([0 * a1, U[2], U[2] * u1 + a1 * p1, u1 * (U[3] + a1 * p1),
                     U[5], U[5] * u2 + a2 * p2, u2 * (U[6] + a2 * p2)])


def Cvec(U):
    a1, r1, u1, p1, a2, r2, u2, p2 = prim(U)
    uI, pI = u2, p1
    return np.array([uI, 0 * uI, -pI, -pI * uI, 0 * uI, pI, pI * uI])


def vmax(U):
    a1, r1, u1, p1, a2, r2, u2, p2 = prim(U)
    c1 = np.sqrt(g1 * (p1 + pinf1) / r1); c2 = np.sqrt(g2 * (p2 + pinf2) / r2)
    return np.maximum(np.abs(u1) + c1, np.abs(u2) + c2)


def salto_nc(UL, UR):
    """int_Phi C dalpha pelo caminho linear em U (alpha é a 1a componente)."""
    dU = UR - UL
    I = sum(w * Cvec(UL + s * dU) for s, w in zip(sq, wq))
    return I * dU[0]


def minmod(a, b, c):
    s = np.sign(a)
    return np.where((np.sign(b) == s) & (np.sign(c) == s), s * np.minimum(np.minimum(np.abs(a), np.abs(b)), np.abs(c)), 0.0)


def escala_positividade(U, dc, margem=0.9):
    """Escala de positividade (Lema 5.1 das notas): um único theta_i por célula, aplicado
    à inclinação inteira, tal que os dois valores de face fiquem no conjunto admissível."""
    theta = np.ones(U.shape[1])
    for sinal in (-1.0, 1.0):
        d = 0.5 * sinal * dc
        for g0, g1 in ((U[0], d[0]), (1.0 - U[0], -d[0]), (U[1], d[1]), (U[4], d[4])):
            neg = g1 < 0
            theta = np.where(neg, np.minimum(theta, margem * g0 / np.where(neg, -g1, 1.0)), theta)
    for _ in range(40):                                  # restrição não linear (energia interna)
        ok = np.ones_like(theta, dtype=bool)
        for sinal in (-1.0, 1.0):
            V = U + 0.5 * sinal * theta * dc
            m1 = V[1]; q1 = V[2]; m2 = V[4]; q2 = V[5]
            e1 = V[3] - 0.5 * q1**2 / m1 - V[0] * pinf1
            e2 = V[6] - 0.5 * q2**2 / m2 - (1 - V[0]) * pinf2
            ok &= (e1 > 0) & (e2 > 0)
        if ok.all():
            break
        theta = np.where(ok, theta, 0.5 * theta)
    return theta


def faces(U, limitador=False):
    Ue = np.concatenate((U[:, -1:], U, U[:, :1]), axis=1)
    dc = 0.5 * (Ue[:, 2:] - Ue[:, :-2])
    if limitador:
        dc = minmod(dc, Ue[:, 2:] - Ue[:, 1:-1], Ue[:, 1:-1] - Ue[:, :-2])
    dc = escala_positividade(U, dc) * dc
    return U - 0.5 * dc, U + 0.5 * dc


def lambda_max(U):
    UL_c, UR_c = faces(U)
    return max(vmax(U).max(), vmax(UL_c).max(), vmax(UR_c).max())


def residuo(U, h, variante, limitador):
    Ue = np.concatenate((U[:, -1:], U, U[:, :1]), axis=1)          # periódico
    dc = 0.5 * (Ue[:, 2:] - Ue[:, :-2])                              # h * s_i (central)
    if limitador:
        dc = minmod(dc, Ue[:, 2:] - Ue[:, 1:-1], Ue[:, 1:-1] - Ue[:, :-2])
    dc = escala_positividade(U, dc) * dc                             # escala (Seção 5.1)
    UL_c = U - 0.5 * dc                                              # u_i^L
    UR_c = U + 0.5 * dc                                              # u_i^R
    # face i+1/2 (i = 0..N-1): estado esquerdo u_i^R, direito u_{i+1}^L
    Lf = UR_c
    Rf = np.roll(UL_c, -1, axis=1)
    lam = np.maximum(vmax(Lf), vmax(Rf))
    F = 0.5 * (fluxo(Lf) + fluxo(Rf)) - 0.5 * lam * (Rf - Lf)
    if variante == "C":                                              # flutuações com as médias
        J = salto_nc(U, np.roll(U, -1, axis=1))
    else:
        J = salto_nc(Lf, Rf)
    Dm = 0.5 * J; Dp = 0.5 * J                                       # repartição central
    Gi = np.zeros_like(U)
    if variante in ("A", "C"):
        dU = UR_c - UL_c
        Gi = sum(w * Cvec(UL_c + s * dU) for s, w in zip(sG, wG)) * dU[0]
    # h dU_i/dt = -(F_{i+1/2}-F_{i-1/2}) - D^-_{i+1/2} - D^+_{i-1/2} - G_i
    return -(F - np.roll(F, 1, axis=1)) - Dm - np.roll(Dp, 1, axis=1) - Gi


def roda(variante, perfil, N=200, cfl=0.1, T=1.0e-3, nmax=10**9, historico=None):
    h = 1.0 / N
    x = (np.arange(N) + 0.5) * h
    p0, u0 = 1.0e5, 100.0
    if perfil == "suave":
        a1 = 0.1 + 0.8 * np.exp(-((x - 0.5) / 0.1) ** 2)
        r1 = 1000.0 + 0 * x; r2 = 1.0 + 0 * x
        lim = False
    else:
        dentro = (x > 0.25) & (x < 0.75)
        a1 = np.where(dentro, 0.8, 0.2)
        r1 = np.where(dentro, 1000.0, 1100.0); r2 = np.where(dentro, 1.0, 2.0)
        lim = False          # inclinação central (Definição 4.1), sem limitador
    U = cons(a1, r1, u0 + 0 * x, p0 + 0 * x, r2, u0 + 0 * x, p0 + 0 * x)
    t = 0.0; n = 0
    while t < T - 1e-15 and n < nmax:
        dt = min(cfl * h / lambda_max(U), T - t)
        U1 = U + dt / h * residuo(U, h, variante, lim)
        U = 0.5 * U + 0.5 * (U1 + dt / h * residuo(U1, h, variante, lim))   # SSP-RK2
        t += dt; n += 1
        if historico is not None:
            q = prim(U)
            historico.append(max(np.abs(q[2] - u0).max(), np.abs(q[6] - u0).max()) / u0)
        if not np.isfinite(U).all():
            break
    a1, r1, u1, p1, a2, r2, u2, p2 = prim(U)
    du = max(np.abs(u1 - u0).max(), np.abs(u2 - u0).max()) / u0
    dp = max(np.abs(p1 - p0).max(), np.abs(p2 - p0).max()) / p0
    return x, U, du, dp


if __name__ == "__main__":
    import os
    np.seterr(all="ignore")
    out = {}
    for perfil in ("suave", "degrau"):
        for v in ("A", "B", "C"):
            hist = []
            x, U, du, dp = roda(v, perfil, historico=hist)
            n_ok = int(np.sum(np.isfinite(hist)))
            h10 = hist[9] if len(hist) > 9 else float("nan")
            print(f"{perfil:7s} {v}: passos={len(hist):4d}  desvio relativo de u apos 10 passos = {h10:.2e}"
                  f"   final: du={du:.2e} dp={dp:.2e}")
            out[f"{perfil}_{v}_hist"] = np.array(hist)
            # perfil após 10 passos
            x10, U10, _, _ = roda(v, perfil, nmax=10)
            q = prim(U10)
            out[f"{perfil}_{v}_x"] = x10; out[f"{perfil}_{v}_u1"] = q[2]; out[f"{perfil}_{v}_p1"] = q[3]
            out[f"{perfil}_{v}_u2"] = q[6]; out[f"{perfil}_{v}_p2"] = q[7]
            out[f"{perfil}_{v}_a1"] = q[0]
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "res_def42.npz"), **out)
