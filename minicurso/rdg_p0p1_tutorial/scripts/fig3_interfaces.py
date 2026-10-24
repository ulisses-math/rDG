# -*- coding: utf-8 -*-
"""Figura 3: Passo 2, os dois estados em cada interface."""
import numpy as np
import matplotlib.pyplot as plt
from estilo_tut import *
from rdg_p0p1 import mesh, sod_exact, sod_exact_average, minmod

N = 24
x, dx = mesh(N)
rbar = sod_exact_average(x, dx, 0.2)
dp = np.zeros(N); dm = np.zeros(N)
dp[:-1] = rbar[1:] - rbar[:-1]; dm[1:] = rbar[1:] - rbar[:-1]
uhat = 0.5 * minmod(dp, dm); uhat[0] = uhat[-1] = 0
xf = x[:-1] + dx / 2                        # interfaces interiores
um = rbar[:-1] + uhat[:-1]                  # u^- (vem da celula da esquerda)
up = rbar[1:] - uhat[1:]                    # u^+ (vem da celula da direita)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.7), gridspec_kw=dict(width_ratios=[1.5, 1]))
for i in range(N):
    a1.plot([x[i] - dx / 2, x[i] + dx / 2], [rbar[i] - uhat[i], rbar[i] + uhat[i]], color=ROXO, lw=1.8)
a1.plot([], [], color=ROXO, lw=1.8, label="reconstrução P1 em cada célula")
a1.plot(xf, um, "v", color=LARANJA, ms=6, label=r"$u^-_{i+1/2}=\bar u_i+\hat u_i$ (limite pela esquerda)")
a1.plot(xf, up, "^", color=TEAL, ms=6, label=r"$u^+_{i+1/2}=\bar u_{i+1}-\hat u_{i+1}$ (limite pela direita)")
a1.vlines(xf, np.minimum(um, up), np.maximum(um, up), color=VERMELHO, lw=1.2, label="salto $u^- - u^+$")
a1.set_xlim(0, 1); a1.set_ylim(0, 1.1); a1.set_xlabel("x"); a1.set_ylabel(r"$\rho$")
a1.set_title("(a) Os dois estados em cada interface")
a1.legend(loc="upper right", fontsize=7)
# zoom: tres interfaces com carater diferente
escolhidas = [np.argmin(np.abs(xf - 0.35)), np.argmin(np.abs(xf - 0.69)), np.argmin(np.abs(xf - 0.85))]
rot = ["região suave\n(salto pequeno)", "contato\n(salto grande)", "choque\n(salto grande)"]
for k, (j, r) in enumerate(zip(escolhidas, rot)):
    a2.bar(k - 0.17, um[j], width=0.32, color=LARANJA, label=r"$u^-$" if k == 0 else None)
    a2.bar(k + 0.17, up[j], width=0.32, color=TEAL, label=r"$u^+$" if k == 0 else None)
    a2.annotate("", xy=(k, up[j]), xytext=(k, um[j]), arrowprops=dict(arrowstyle="<->", color=VERMELHO, lw=1.2))
    a2.text(k + 0.36, 0.5 * (um[j] + up[j]), "salto\n%.3f" % (um[j] - up[j]), color=VERMELHO, fontsize=8, va="center")
a2.set_xticks([0, 1, 2]); a2.set_xticklabels([r + "\n$x=%.3f$" % xf[j] for j, r in zip(escolhidas, rot)], fontsize=8)
a2.set_ylim(0, 1.15); a2.set_xlim(-0.6, 2.8); a2.set_ylabel(r"$\rho$")
a2.set_title("(b) Zoom em três interfaces: o salto define o problema de Riemann")
a2.legend(loc="upper right")
suptitle(fig, "Passo 2 — Estados nas interfaces")
fig.tight_layout(w_pad=2, rect=(0, 0, 1, 0.95))
salvar(fig, "fig3_interfaces.png")
