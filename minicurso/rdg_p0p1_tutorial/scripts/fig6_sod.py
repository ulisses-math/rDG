# -*- coding: utf-8 -*-
"""Figura 6: FV padrao (P0) contra rDG(P0P1) na mesma malha, tubo de Sod."""
import os
import numpy as np
import matplotlib.pyplot as plt
from estilo_tut import *
from rdg_p0p1 import sod_exact, GAMMA

d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.npz"))
xx = np.linspace(0, 1, 2001); rex, vex, pex = sod_exact(xx, 0.2); eex = pex / ((GAMMA - 1) * rex)
N = 100
x = d[f"sod_{N}_1.0_x"]
r1, v1, p1 = d[f"sod_{N}_1.0_rho"], d[f"sod_{N}_1.0_v"], d[f"sod_{N}_1.0_p"]
r0, v0, p0 = d[f"sodP0_{N}_rho"], d[f"sodP0_{N}_v"], d[f"sodP0_{N}_p"]
fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.6)); axs = axs.ravel()
nomes = [r"Densidade $\rho$", r"Velocidade $v$", r"Pressão $p$", r"Energia interna $e=p/((\gamma-1)\rho)$"]
for a, qe, q0, q1, nome in zip(axs, (rex, vex, pex, eex), (r0, v0, p0, p0 / ((GAMMA - 1) * r0)), (r1, v1, p1, p1 / ((GAMMA - 1) * r1)), nomes):
    a.plot(xx, qe, "k-", lw=1.3, label="solução exata")
    a.plot(x, q0, "s", color=CINZA, ms=3.5, label="FV padrão, P0 (1ª ordem)")
    a.plot(x, q1, "o", color=ROXO, ms=3.5, label="rDG(P0P1) + minmod (2ª ordem)")
    a.set_title(nome); a.set_xlabel("x"); a.set_xlim(0, 1)
axs[0].legend(loc="upper right")
suptitle(fig, "Tubo de Sod, t = 0,2, mesma malha (N = 100) e mesmo fluxo HLLC: só muda a reconstrução")
fig.tight_layout(rect=(0, 0, 1, 0.95))
salvar(fig, "fig6_sod.png")
