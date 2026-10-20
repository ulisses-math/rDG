# -*- coding: utf-8 -*-
"""Figura: os três problemas de Riemann de Burgers (choque, rarefação, rarefação transônica)."""
import numpy as np
import matplotlib.pyplot as plt
from estilo import *
from fluxos import exato_riemann_b

casos = [(1.0, -0.5, "choque: $u_L=1>u_R=-0.5$, $s=(u_L+u_R)/2=0.25$"),
         (0.3, 1.0, "rarefação: $0<u_L=0.3<u_R=1$"),
         (-0.6, 1.0, "rarefação transônica: $u_L=-0.6<0<u_R=1$")]
fig, axs = plt.subplots(2, 3, figsize=(11, 5.2), gridspec_kw=dict(height_ratios=[1.25, 1]))
T = 1.0
for k, (uL, uR, tit) in enumerate(casos):
    ax = axs[0, k]
    for x0 in np.arange(-2.0, 2.01, 0.2):
        u = uL if x0 < 0 else uR
        tmax = T * 1.6
        if uL > uR:                      # a característica morre ao tocar o choque
            s = 0.5 * (uL + uR)
            if (x0 < 0 and u > s) or (x0 > 0 and u < s):
                tmax = min(tmax, x0 / (s - u))
        xs = np.array([x0, x0 + u * tmax])
        ax.plot(xs, [0, tmax], color=AZULCLARO if x0 < 0 else VERDE, lw=0.9)
    if uL > uR:
        s = 0.5 * (uL + uR)
        ax.plot([0, s * T * 1.6], [0, T * 1.6], color=VERMELHO, lw=2.2, label="choque, $dx/dt=s$")
    else:
        for u in np.linspace(uL, uR, 7):
            ax.plot([0, u * T * 1.6], [0, T * 1.6], color=LARANJA, lw=1.3)
        ax.plot([], [], color=LARANJA, lw=1.3, label="leque $u=x/t$")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlim(-2, 2); ax.set_ylim(0, T * 1.6); ax.set_title(tit, fontsize=9.5)
    ax.set_ylabel("$t$"); ax.set_xticklabels([])
    ax.legend(loc="upper left", fontsize=8)
    ax = axs[1, k]
    x = np.linspace(-2, 2, 1001)
    ax.plot(x, exato_riemann_b(x, 0.0, uL, uR), color=CINZA, lw=1.1, ls="--", label="$t=0$")
    ax.plot(x, exato_riemann_b(x, T, uL, uR), color=AZUL, lw=1.8, label="$t=1$")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    uf = exato_riemann_b(np.array([0.0]), T, uL, uR)[0]
    ax.plot([0], [uf], "o", color=LARANJA, ms=6, zorder=5, label="valor na face: $u^{RP}(0)=%.2g$" % uf)
    ax.set_xlim(-2, 2); ax.set_ylim(-1.0, 1.3); ax.set_xlabel("$x$"); ax.set_ylabel("$u$")
    ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(h_pad=0.6)
salvar(fig, "riemann_burgers.pdf")
