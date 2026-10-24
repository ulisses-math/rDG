# -*- coding: utf-8 -*-
"""Figura 7: convergencia com a malha (Sod e onda de densidade suave) e o efeito do passo de tempo."""
import os
import numpy as np
import matplotlib.pyplot as plt
from estilo_tut import *
from rdg_p0p1 import sod_exact

d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.npz"))
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8))
a = axs[0]
N = d["conv_wave_N"]
a.loglog(N, d["conv_wave_zero"], "s-", color=CINZA, label="P0 (sem reconstrução)")
a.loglog(N, d["conv_wave_minmod"], "o-", color=ROXO, label="rDG(P0P1), minmod")
a.loglog(N, d["conv_wave_none"], "^-", color=AZUL2, label="rDG(P0P1), sem limitador")
a.loglog(N, 1.2 * d["conv_wave_zero"][0] * (N[0] / N) ** 1, ":", color="0.4", label=r"referência $\propto N^{-1}$")
a.loglog(N, 1.2 * d["conv_wave_none"][0] * (N[0] / N) ** 2, "--", color="0.4", label=r"referência $\propto N^{-2}$")
a.set_xticks(N); a.set_xticklabels([str(n) for n in N]); a.minorticks_off()
a.set_xlabel("N (células)"); a.set_ylabel(r"erro $L^1$ em $\rho$"); a.set_title("(a) Solução suave: onda de densidade, t = 1")
a.legend(loc="lower left", fontsize=7)
b = axs[1]
N = d["conv_sod_N"]
b.loglog(N, d["conv_sod_zero"], "s-", color=CINZA, label="P0 (sem reconstrução)")
b.loglog(N, d["conv_sod_minmod"], "o-", color=ROXO, label="rDG(P0P1), minmod")
b.loglog(N, 1.3 * d["conv_sod_zero"][0] * (N[0] / N) ** 0.5, ":", color="0.4", label=r"referência $\propto N^{-1/2}$")
b.loglog(N, 1.3 * d["conv_sod_minmod"][0] * (N[0] / N) ** 1, "--", color="0.4", label=r"referência $\propto N^{-1}$")
b.set_xticks(N); b.set_xticklabels([str(n) for n in N]); b.minorticks_off()
b.set_xlabel("N (células)"); b.set_ylabel(r"erro $L^1$ em $\rho$"); b.set_title("(b) Com descontinuidades: Sod, t = 0,2")
b.legend(loc="lower left", fontsize=7)
c = axs[2]
xx = np.linspace(0, 1, 2001); rex, _, _ = sod_exact(xx, 0.2)
c.plot(xx, rex, "k-", lw=1.2, label="exata")
c.plot(d["sod_100_1.0_x"], d["sod_100_1.0_rho"], "o-", color=AZUL, ms=3, lw=0.8, label="CFL = 1 (24 passos)")
c.plot(d["sod_100_5.0_x"], d["sod_100_5.0_rho"], "s-", color=VERMELHO, ms=3, lw=0.8, label="CFL = 5 (5 passos)")
c.set_xlim(0.55, 0.95); c.set_ylim(0.08, 0.5); c.set_xlabel("x"); c.set_ylabel(r"$\rho$")
c.set_title("(c) O preço do passo grande (N = 100, zoom)")
c.legend(loc="upper right", fontsize=7.5)
suptitle(fig, "Convergência com a malha e escolha do passo de tempo")
fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.94))
salvar(fig, "fig7_convergencia.png")
