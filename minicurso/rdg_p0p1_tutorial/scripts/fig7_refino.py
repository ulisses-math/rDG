# -*- coding: utf-8 -*-
"""Figura 7: refino da malha, P0 e rDG(P0P1), com zoom nas tres ondas."""
import os
import numpy as np
import matplotlib.pyplot as plt
from estilo_tut import *
from rdg_p0p1 import sod_exact

d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.npz"))
xx = np.linspace(0, 1, 4001); rex, _, _ = sod_exact(xx, 0.2)
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
zonas = [((0.2, 0.55), "(a) rarefação"), ((0.6, 0.78), "(b) contato"), ((0.78, 0.92), "(c) choque")]
for a, ((x0, x1), tit) in zip(axs, zonas):
    a.plot(xx, rex, "k-", lw=1.4, label="exata")
    for N, mk in ((100, "s"), (400, ".")):
        a.plot(d[f"sodP0_{N}_x"], d[f"sodP0_{N}_rho"], mk, color=CINZA, ms=4 if N == 100 else 3, label=f"P0, N = {N}")
    for N, mk in ((100, "o"), (400, ".")):
        a.plot(d[f"sod_{N}_1.0_x"], d[f"sod_{N}_1.0_rho"], mk, color=ROXO, ms=4 if N == 100 else 3, label=f"rDG(P0P1), N = {N}")
    m = (xx > x0) & (xx < x1)
    a.set_xlim(x0, x1); a.set_ylim(rex[m].min() - 0.05, rex[m].max() + 0.05)
    a.set_xlabel("x"); a.set_ylabel(r"$\rho$"); a.set_title(tit + ": zoom na densidade")
axs[0].legend(loc="lower left", fontsize=7.5)
suptitle(fig, "Refinando a malha: o FV padrão precisa de 4 vezes mais células para igualar o rDG(P0P1)")
fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.94))
salvar(fig, "fig7_refino.png")
