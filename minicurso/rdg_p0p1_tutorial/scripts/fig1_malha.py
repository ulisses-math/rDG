# -*- coding: utf-8 -*-
"""Figura 1: a malha, a media celular e a representacao P0."""
import numpy as np
import matplotlib.pyplot as plt
from estilo_tut import *
from rdg_p0p1 import mesh, sod_exact, sod_exact_average

N = 20
x, dx = mesh(N)
xx = np.linspace(0, 1, 2001)
rex, _, _ = sod_exact(xx, 0.2)
rbar = sod_exact_average(x, dx, 0.2)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
a1.plot(xx, rex, "k-", lw=1.3, label=r"solução $u(x)$ (contínua)")
a1.bar(x, rbar, width=dx, color=AZUL2, alpha=0.28, edgecolor=AZUL, lw=0.8, label=r"média celular $\bar u_i$ (P0)")
a1.plot(x, rbar, "o", color=LARANJA, ms=4, label=r"o único número guardado por célula")
for xf in np.arange(0, 1.0001, dx):
    a1.axvline(xf, color="0.75", lw=0.5, zorder=0)
i = 12
a1.annotate("", xy=(x[i] - dx / 2, 1.12), xytext=(x[i] + dx / 2, 1.12), arrowprops=dict(arrowstyle="<->", color=VERMELHO, lw=1.2))
a1.text(x[i], 1.14, r"$\Delta x$", color=VERMELHO, ha="center", fontsize=9)
a1.text(x[i] - dx / 2, -0.07, r"$x_{i-1/2}$", ha="right", fontsize=8, color=VERMELHO)
a1.text(x[i] + dx / 2, -0.07, r"$x_{i+1/2}$", ha="left", fontsize=8, color=VERMELHO)
a1.text(x[i], -0.07, r"$x_i$", ha="center", fontsize=8, color=VERMELHO)
a1.axvline(x[i] - dx / 2, color=VERMELHO, lw=1.2); a1.axvline(x[i] + dx / 2, color=VERMELHO, lw=1.2)
a1.plot([x[i]], [0], "v", color=VERMELHO, ms=5, clip_on=False)
a1.set_xlim(0, 1); a1.set_ylim(-0.15, 1.25); a1.set_xlabel("x"); a1.set_ylabel(r"$\rho$")
a1.set_title(r"Malha com $N=20$ células e a representação P0 (densidade de Sod em $t=0{,}2$)")
a1.legend(loc="upper right")
# zoom numa celula: a media como "area igual"
j = 13
xl, xr = x[j] - dx / 2, x[j] + dx / 2
m = (xx >= xl) & (xx <= xr)
a2.fill_between(xx[m], rbar[j], rex[m], where=rex[m] > rbar[j], color=VERDE, alpha=0.35, lw=0, label="sobra (acima da média)")
a2.fill_between(xx[m], rbar[j], rex[m], where=rex[m] < rbar[j], color=VERMELHO, alpha=0.35, lw=0, label="falta (abaixo da média)")
a2.plot(xx[m], rex[m], "k-", lw=1.5, label=r"$u(x)$ dentro da célula")
a2.hlines(rbar[j], xl, xr, color=AZUL, lw=2.2, label=r"$\bar u_i$: mesma área")
a2.axvline(xl, color="0.6", lw=0.8); a2.axvline(xr, color="0.6", lw=0.8)
a2.set_xlim(xl - 0.004, xr + 0.004); a2.set_xlabel("x")
a2.set_title(r"Zoom na célula $i=%d$: a média conserva a área" % j)
a2.legend(loc="upper right", fontsize=7)
fig.tight_layout(w_pad=2)
salvar(fig, "fig1_malha.png")
