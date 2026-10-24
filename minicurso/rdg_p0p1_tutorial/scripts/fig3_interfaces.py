# -*- coding: utf-8 -*-
"""Figura 3: Passo 2, os dois estados em cada interface."""
import numpy as np
import matplotlib.pyplot as plt
from estilo_tut import *
from rdg_p0p1 import mesh, sod_exact, sod_exact_average, minmod

N = 16
x, dx = mesh(N)
rbar = sod_exact_average(x, dx, 0.2)
xx = np.linspace(0, 1, 2001); rex, _, _ = sod_exact(xx, 0.2)
dp = np.zeros(N); dm = np.zeros(N)
dp[:-1] = rbar[1:] - rbar[:-1]; dm[1:] = rbar[1:] - rbar[:-1]
uhat = 0.5 * minmod(dp, dm); uhat[0] = uhat[-1] = 0
xf = x[:-1] + dx / 2
um = rbar[:-1] + uhat[:-1]
up = rbar[1:] - uhat[1:]
jump0 = np.abs(rbar[1:] - rbar[:-1])        # salto entre medias (P0)
jump1 = np.abs(um - up)                     # salto entre valores reconstruidos

fig = plt.figure(figsize=(13, 8.2))
gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.75, 1.05], hspace=0.6, wspace=0.22)
# (a) reconstrucao, celulas alternadas em duas cores para separar visualmente
a1 = fig.add_subplot(gs[0, :])
a1.plot(xx, rex, "k--", lw=0.9, label="solução exata")
for i in range(N):
    cor = ROXO if i % 2 == 0 else AZUL2
    a1.plot([x[i] - dx / 2, x[i] + dx / 2], [rbar[i] - uhat[i], rbar[i] + uhat[i]], color=cor, lw=3, solid_capstyle="butt")
for j in range(len(xf)):
    a1.axvline(xf[j], color="0.85", lw=0.6, zorder=0)
    if jump1[j] > 0.005:
        a1.plot([xf[j], xf[j]], [min(um[j], up[j]), max(um[j], up[j])], color=VERMELHO, lw=3, zorder=4)
a1.plot([], [], color=ROXO, lw=3, label="reta P1 das células pares")
a1.plot([], [], color=AZUL2, lw=3, label="reta P1 das células ímpares")
a1.plot([], [], color=VERMELHO, lw=3, label=r"salto $u^- - u^+$ na interface")
a1.set_xticks(x); a1.set_xticklabels([str(i) for i in range(N)], fontsize=7)
a1.set_xlim(0, 1); a1.set_ylim(0, 1.12); a1.set_xlabel("índice da célula $i$"); a1.set_ylabel(r"$\rho$")
a1.set_title("(a) Cada célula tem a sua reta; onde duas retas vizinhas não se encontram há um salto (vermelho)")
a1.legend(loc="upper right", fontsize=8)
# (b) tamanho do salto em cada interface: P0 x P0P1
a2 = fig.add_subplot(gs[1, :])
w = 0.38 * dx
a2.bar(xf - w / 2, jump0, width=w, color=CINZA, label=r"sem reconstrução (P0): $|\bar u_{i+1}-\bar u_i|$")
a2.bar(xf + w / 2, jump1, width=w, color=VERMELHO, label=r"com reconstrução (P0P1): $|u^-_{i+1/2}-u^+_{i+1/2}|$")
a2.set_xticks(xf); a2.set_xticklabels([r"%d|%d" % (i, i + 1) for i in range(N - 1)], fontsize=7)
a2.set_xlim(0, 1); a2.set_xlabel("interface entre as células"); a2.set_ylabel("salto")
a2.set_title("(b) Tamanho do salto em cada interface: a reconstrução encolhe os saltos onde a solução é suave")
a2.legend(loc="upper left", fontsize=8)
# (c, d) zooms em duas interfaces
casos = [(5, "(c) Zoom na interface 5|6 (rarefação): salto pequeno"),
         (10, "(d) Zoom na interface 10|11 (contato): salto grande")]
for k, (j, tit) in enumerate(casos):
    a = fig.add_subplot(gs[2, k])
    i = j
    m = (xx >= x[i] - dx / 2) & (xx <= x[i + 1] + dx / 2)
    a.plot(xx[m], rex[m], "k--", lw=1, label="solução exata")
    for c, cor in ((i, ROXO), (i + 1, AZUL2)):
        a.plot([x[c] - dx / 2, x[c] + dx / 2], [rbar[c] - uhat[c], rbar[c] + uhat[c]], color=cor, lw=3.5, solid_capstyle="butt")
        a.plot([x[c]], [rbar[c]], "o", color=cor, ms=6)
    a.plot([xf[j]], [um[j]], "o", color=LARANJA, ms=10, mec="k", mew=0.6, zorder=5, label=r"$u^-$ (fim da reta da esquerda)")
    a.plot([xf[j]], [up[j]], "s", color=TEAL, ms=9, mec="k", mew=0.6, zorder=5, label=r"$u^+$ (início da reta da direita)")
    a.plot([xf[j], xf[j]], [min(um[j], up[j]), max(um[j], up[j])], color=VERMELHO, lw=3, zorder=4)
    a.axvline(xf[j], color="0.6", lw=0.8, ls=":")
    lo = min(rbar[i], rbar[i + 1], um[j], up[j]); hi = max(rbar[i], rbar[i + 1], um[j], up[j])
    pad = 0.25 * (hi - lo) + 0.02
    a.set_ylim(lo - pad, hi + 1.6 * pad)
    a.set_xlim(x[i] - dx / 2, x[i + 1] + dx / 2)
    a.set_xticks([x[i], xf[j], x[i + 1]])
    a.set_xticklabels(["centro da\ncélula %d" % i, "interface\n%d|%d" % (i, i + 1), "centro da\ncélula %d" % (i + 1)], fontsize=8)
    txt = (r"$\bar u_{%d}=%.3f$,  $\hat u_{%d}=%+.3f$" % (i, rbar[i], i, uhat[i]) + "\n" +
           r"$\bar u_{%d}=%.3f$,  $\hat u_{%d}=%+.3f$" % (i + 1, rbar[i + 1], i + 1, uhat[i + 1]) + "\n" +
           r"$u^-=%.3f$,  $u^+=%.3f$" % (um[j], up[j]) + "\n" +
           r"salto $=%.3f$  (P0 daria $%.3f$)" % (um[j] - up[j], rbar[i] - rbar[i + 1]))
    a.text(0.98, 0.97, txt, transform=a.transAxes, ha="right", va="top", fontsize=8,
           bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"))
    a.set_title(tit, fontsize=9.5); a.set_ylabel(r"$\rho$")
    a.legend(loc="lower left", fontsize=7.5)
suptitle(fig, "Passo 2 — Estados nas interfaces (densidade de Sod em t = 0,2, malha de N = 16 células)")
salvar(fig, "fig3_interfaces.png")
