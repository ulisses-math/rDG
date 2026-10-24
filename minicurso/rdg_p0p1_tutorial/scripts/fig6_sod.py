# -*- coding: utf-8 -*-
"""Figura 6: resultado em Sod com tres malhas (CFL = 1)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from estilo_tut import *
from rdg_p0p1 import sod_exact, GAMMA

d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.npz"))
xx = np.linspace(0, 1, 2001); rex, vex, pex = sod_exact(xx, 0.2); eex = pex / ((GAMMA - 1) * rex)
fig, axs = plt.subplots(2, 2, figsize=(11, 6.4)); axs = axs.ravel()
nomes = [r"Densidade $\rho$", r"Velocidade $v$", r"Pressão $p$", r"Energia interna $e=p/((\gamma-1)\rho)$"]
for a, q, nome in zip(axs, (rex, vex, pex, eex), nomes):
    a.plot(xx, q, "k-", lw=1.2, label="exata"); a.set_title(nome); a.set_xlabel("x"); a.set_xlim(0, 1)
for N, cor, mk in ((50, LARANJA, "o"), (100, VERDE, "s"), (400, AZUL, ".")):
    x = d[f"sod_{N}_1.0_x"]; r = d[f"sod_{N}_1.0_rho"]; v = d[f"sod_{N}_1.0_v"]; p = d[f"sod_{N}_1.0_p"]
    e = p / ((GAMMA - 1) * r)
    for a, q in zip(axs, (r, v, p, e)):
        a.plot(x, q, mk, color=cor, ms=3.2 if N < 400 else 2.5, lw=0.8, ls="-" if N == 400 else "none", label=f"N = {N}", alpha=0.9)
axs[0].legend(loc="upper right"); 
for a, (x0, x1, lab) in zip(axs, ((0.25, 0.5, "rarefação"), (0.6, 0.75, "contato"), (0.8, 0.9, "choque"), (0.6, 0.75, "contato"))):
    a.axvspan(x0, x1, color=AMARELO, alpha=0.12, lw=0); a.text(0.5 * (x0 + x1), a.get_ylim()[0] + 0.08 * (a.get_ylim()[1] - a.get_ylim()[0]), lab, ha="center", fontsize=8, color="0.35")
suptitle(fig, "Resultado — rDG(P0P1) + HLLC + minmod + BDF2 no tubo de Sod (t = 0,2, CFL = 1)")
fig.tight_layout(rect=(0, 0, 1, 0.95))
salvar(fig, "fig6_sod.png")
