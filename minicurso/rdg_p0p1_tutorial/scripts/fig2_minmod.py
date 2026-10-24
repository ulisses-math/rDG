# -*- coding: utf-8 -*-
"""Figura 2: Passo 1, reconstrucao com minmod, numa solucao com regioes suaves e saltos."""
import numpy as np
import matplotlib.pyplot as plt
from estilo_tut import *
from rdg_p0p1 import mesh, sod_exact, sod_exact_average, minmod

N = 24
x, dx = mesh(N)
rbar = sod_exact_average(x, dx, 0.2)
xx = np.linspace(0, 1, 2001); rex, _, _ = sod_exact(xx, 0.2)
dp = np.zeros(N); dm = np.zeros(N)
dp[:-1] = rbar[1:] - rbar[:-1]; dm[1:] = rbar[1:] - rbar[:-1]
uhat = 0.5 * minmod(dp, dm); uhat[0] = uhat[-1] = 0
ucen = 0.25 * (dp + dm)
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.7), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
a = axs[0]
w = 0.38 * dx
a.bar(x - w / 2, dp, width=w, color=AZUL2, label=r"$\Delta_{+i}=\bar u_{i+1}-\bar u_i$")
a.bar(x + w / 2, dm, width=w, color=LARANJA, label=r"$\Delta_{-i}=\bar u_i-\bar u_{i-1}$")
a.axhline(0, color="k", lw=0.8)
a.set_title("(a) As duas diferenças em cada célula"); a.set_xlabel("x"); a.set_xlim(0, 1)
a.legend(loc="lower left")
a = axs[1]
a.plot(x, ucen, "s", color=CINZA, ms=4, label=r"sem limitador: $\frac{1}{4}(\Delta_+ + \Delta_-)$")
a.plot(x, uhat, "o", color=ROXO, ms=5, label=r"minmod: $\hat u_i=\frac{1}{2}\,\mathrm{minmod}(\Delta_+,\Delta_-)$")
zer = (uhat == 0) & ((dp != 0) | (dm != 0))
a.plot(x[zer], uhat[zer], "x", color=VERMELHO, ms=9, mew=1.8, label="zerado: sinais discordam ou vizinho constante")
a.axhline(0, color="k", lw=0.8)
a.set_title(r"(b) A inclinação escolhida, $\hat u_i$"); a.set_xlabel("x"); a.set_xlim(0, 1)
a.legend(loc="lower left", fontsize=7)
a = axs[2]
a.plot(xx, rex, "k--", lw=1, label="solução exata")
a.bar(x, rbar, width=dx, color=AZUL2, alpha=0.18, edgecolor="0.7", lw=0.5, label=r"P0: médias $\bar u_i$")
for i in range(N):
    a.plot([x[i] - dx / 2, x[i] + dx / 2], [rbar[i] - uhat[i], rbar[i] + uhat[i]], color=ROXO, lw=2)
a.plot([], [], color=ROXO, lw=2, label=r"P1: $\bar u_i+\hat u_i\,\xi$")
a.plot(x - dx / 2, rbar - uhat, "<", color=ROXO, ms=3.5); a.plot(x + dx / 2, rbar + uhat, ">", color=ROXO, ms=3.5)
a.set_title("(c) A reconstrução linear por célula"); a.set_xlabel("x"); a.set_xlim(0, 1); a.set_ylim(0, 1.1)
a.legend(loc="upper right")
suptitle(fig, "Passo 1 — Reconstrução minmod (densidade de Sod em t = 0,2, N = 24)")
fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.95))
salvar(fig, "fig2_minmod.png")
