# -*- coding: utf-8 -*-
"""Figura: o 'botão' da dissipação Q na advecção linear (forma central + dissipação)."""
import numpy as np
import matplotlib.pyplot as plt
from estilo import *

a, nu, N, L, tf = 1.0, 0.5, 200, 2.0, 0.5
h = L / N; dt = nu * h / a
x = (np.arange(N) + 0.5) * h
u0 = np.where((x > 0.2) & (x < 0.6), 1.0, 0.0) + np.exp(-((x - 1.3) / 0.1) ** 2)


def exata(x, t):
    xx = (x - a * t) % L
    return np.where((xx > 0.2) & (xx < 0.6), 1.0, 0.0) + np.exp(-((xx - 1.3) / 0.1) ** 2)


def roda(Q, u0, nsteps):
    u = u0.copy()
    for _ in range(nsteps):
        ue = np.concatenate(([u[-1]], u, [u[0]]))
        uL, uR = ue[:-1], ue[1:]
        F = 0.5 * a * (uL + uR) - 0.5 * Q * (uR - uL)
        u = u - dt / h * (F[1:] - F[:-1])
    return u


casos = [(nu * a, "$Q=\\nu|a|$ (Lax–Wendroff)", VERMELHO, "-"),
         (a, "$Q=|a|$ (upwind)", AZUL, "-"),
         (0.5 * (a + h / dt), "$Q=\\frac{1}{2}(|a|+h/\\Delta t)$", VERDE, "-"),
         (h / dt, "$Q=h/\\Delta t$ (Lax–Friedrichs)", LARANJA, "-")]
nsteps = int(round(tf / dt))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax1.plot(x, exata(x, tf), "k--", lw=1.1, label="exata, $t=%.1f$" % tf)
for Q, rot, cor, ls in casos:
    ax1.plot(x, roda(Q, u0, nsteps), color=cor, ls=ls, lw=1.4, label=rot)
ax1.set_xlabel("$x$"); ax1.set_ylabel("$u$"); ax1.set_ylim(-0.35, 1.3); ax1.set_xlim(0, L)
ax1.set_title("Advecção, $a=1$, $N=200$, $\\nu=a\\Delta t/h=0.5$: mesmo esquema, só muda $Q$")
ax1.legend(loc="upper left", ncol=1)

th = np.linspace(0, np.pi, 400)
for Q, rot, cor, ls in [(0.3 * a, "$Q=0.3|a|<\\nu|a|$: instável", CINZA, ":")] + casos + \
        [(1.3 * h / dt, "$Q=1.3\\,h/\\Delta t$: instável", CINZA, "--")]:
    mu = dt * Q / h
    g = np.abs(1 - 1j * nu * np.sin(th) + mu * (np.cos(th) - 1))
    ax2.plot(th, g, color=cor, ls=ls, lw=1.4, label=rot)
ax2.axhline(1, color="k", lw=0.8)
ax2.set_xlabel("$\\theta=kh$"); ax2.set_ylabel("$|g(\\theta)|$"); ax2.set_ylim(0, 1.6); ax2.set_xlim(0, np.pi)
ax2.set_title("Fator de amplificação: estável se $\\nu|a|\\leq Q\\leq h/\\Delta t$")
ax2.legend(loc="lower left", fontsize=7.8)
fig.tight_layout(w_pad=1.5)
salvar(fig, "fluxo_dissipacao.pdf")
