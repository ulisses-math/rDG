# -*- coding: utf-8 -*-
"""Figura 2: Euler com energia interna (não conservativa): convergência para choques errados."""
import os
import numpy as np
import matplotlib.pyplot as plt
from estilo import *

aqui = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(aqui, "res_forte_6400.npz"))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
x = d["x"]
ax1.plot(x, d["rex"], "k-", lw=1.2, label="exata")
ax1.plot(x, d["rc"], color=AZUL, lw=1.5, label="energia total (conservativa)")
ax1.plot(x, d["r1"], color=VERMELHO, lw=1.5, label="energia interna, caminho I")
ax1.plot(x, d["r2"], color=LARANJA, lw=1.5, ls="--", label="energia interna, caminho II")
ax1.set_xlim(0.55, 0.82); ax1.set_ylim(0, 12)
ax1.set_xlabel("$x$"); ax1.set_ylabel(r"$\rho$")
ax1.set_title(r"Choque forte ($p_L/p_R=1000$), $N=6400$, $t=0.012$: densidade")
ax1.legend(loc="upper left", fontsize=8)
Ns = [400, 1600, 6400]
E = {k: [] for k in ("rc", "r1", "r2", "dif")}
for N in Ns:
    r = np.load(os.path.join(aqui, f"res_forte_{N}.npz")); h = 1.0 / N
    E["rc"].append(h * np.sum(np.abs(r["rc"] - r["rex"])))
    E["r1"].append(h * np.sum(np.abs(r["r1"] - r["rex"])))
    E["r2"].append(h * np.sum(np.abs(r["r2"] - r["rex"])))
    E["dif"].append(h * np.sum(np.abs(r["r1"] - r["r2"])))
ax2.loglog(Ns, E["rc"], "o-", color=AZUL, label="erro, energia total")
ax2.loglog(Ns, E["r1"], "s-", color=VERMELHO, label="erro, energia interna (I)")
ax2.loglog(Ns, E["r2"], "^--", color=LARANJA, label="erro, energia interna (II)")
ax2.loglog(Ns, E["dif"], "d:", color=CINZA, label="diferença entre I e II")
ax2.set_xticks(Ns); ax2.set_xticklabels([str(n) for n in Ns]); ax2.minorticks_off()
ax2.set_xlabel("$N$ (células)"); ax2.set_ylabel(r"$\|\rho_h-\rho\|_{L^1}$")
ax2.set_title("refinar não corrige: cada caminho tem o seu limite")
ax2.legend(loc="lower left", fontsize=8)
fig.tight_layout(w_pad=2)
salvar(fig, "energia_interna.pdf")
