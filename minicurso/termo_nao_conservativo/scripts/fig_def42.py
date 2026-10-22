# -*- coding: utf-8 -*-
"""Figura 3: teste de Abgrall para a Definição 4.2 (variantes A, B, C)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from estilo import *

aqui = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(aqui, "res_def42.npz"))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
x = d["suave_A_x"]
rot = {"A": "A: Definição 4.2 ($F$, $D^\\pm$ e $G_i$)", "B": "B: sem $G_i$", "C": "C: $D^\\pm$ com as médias"}
cor = {"A": AZUL, "B": VERMELHO, "C": LARANJA}
for v in ("A", "B", "C"):
    ax1.plot(x, (d[f"suave_{v}_u2"] - 100.0) / 100.0, color=cor[v], lw=1.6, ls="-" if v != "C" else "--", label=rot[v])
axb = ax1.twinx()
axb.fill_between(x, d["suave_A_a1"], color="0.85", lw=0, zorder=0)
axb.set_ylim(0, 2.2); axb.set_yticks([]); axb.grid(False)
ax1.set_zorder(axb.get_zorder() + 1); ax1.patch.set_visible(False)
ax1.set_xlabel("$x$  (sombra cinza: perfil de $\\alpha_1$)"); ax1.set_ylabel(r"$(u_2-u_0)/u_0$")
ax1.set_title(r"$p$ e $u$ uniformes, $\alpha_1$ variável: velocidade do gás após 10 passos")
ax1.legend(loc="upper left", fontsize=8)
for v in ("A", "B", "C"):
    for perfil, ls in (("suave", "-"), ("degrau", "--")):
        h = d[f"{perfil}_{v}_hist"]
        n = np.arange(1, len(h) + 1)
        ok = np.isfinite(h)
        ax2.semilogy(n[ok], np.maximum(h[ok], 1e-16), color=cor[v], ls=ls, lw=1.5,
                     label=v if perfil == "suave" else None)
ax2.set_xscale("log")
ax2.set_xlabel("passo de tempo"); ax2.set_ylabel(r"$\max|u-u_0|/u_0$")
ax2.set_title("histórico do desvio (contínua: $\\alpha_1$ suave; tracejada: degrau)", fontsize=9.5)
ax2.legend(loc="center right", fontsize=8)
ax2.set_ylim(1e-16, 1e2)
fig.tight_layout(w_pad=2)
salvar(fig, "def42_abgrall.pdf")
