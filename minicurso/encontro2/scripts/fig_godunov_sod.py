# -*- coding: utf-8 -*-
"""Figura: método de Godunov (solver exato) no tubo de Sod, N=100 e 400."""
import numpy as np
import matplotlib.pyplot as plt
from estilo import *
from fluxos import *

t = 0.2
xx = np.linspace(0, 1, 2001)
rex, uex, pex = amostrar((xx - 0.5) / t, *SOD[0], *SOD[1])
eex = pex / ((GAMMA - 1) * rex)
fig, axs = plt.subplots(2, 2, figsize=(11, 5.6))
axs = axs.ravel()
for ax, q, nome in zip(axs, (rex, uex, pex, eex), ("densidade $\\rho$", "velocidade $u$", "pressão $p$", "energia interna $e=p/((\\gamma-1)\\rho)$")):
    ax.plot(xx, q, "k-", lw=1.0, label="exata")
    ax.set_title(nome); ax.set_xlabel("$x$"); ax.set_xlim(0, 1)
for N, cor, mk in ((100, LARANJA, "o"), (400, AZUL, ".")):
    x, h = malha(0.0, 1.0, N)
    (r, u, p), n = avancar_e(riemann_inicial(x, 0.5, *SOD), x, t, godunov_e, cfl=0.8)
    e = p / ((GAMMA - 1) * r)
    for ax, q in zip(axs, (r, u, p, e)):
        ax.plot(x, q, mk, color=cor, ms=3.5 if N == 100 else 3, lw=0.8, label="Godunov exato, $N=%d$" % N, alpha=0.95)
axs[0].legend(loc="upper right"); axs[3].legend(loc="upper left")
axs[0].annotate("contato: borrado", (0.70, 0.30), xytext=(0.58, 0.55), fontsize=9, arrowprops=dict(arrowstyle="-|>", color="0.3"))
axs[0].annotate("choque: 2–3 células", (0.85, 0.20), xytext=(0.74, 0.75), fontsize=9, arrowprops=dict(arrowstyle="-|>", color="0.3"))
axs[0].annotate("rarefação", (0.37, 0.70), xytext=(0.18, 0.45), fontsize=9, arrowprops=dict(arrowstyle="-|>", color="0.3"))
fig.suptitle("Tubo de Sod em $t=0.2$: Godunov de primeira ordem com solver de Riemann exato, CFL $=0.8$", y=1.0)
fig.tight_layout()
salvar(fig, "godunov_sod.pdf")
