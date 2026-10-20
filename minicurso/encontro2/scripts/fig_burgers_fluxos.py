# -*- coding: utf-8 -*-
"""Figura: Godunov, Roe (sem e com correção de entropia) e Rusanov em Burgers."""
import numpy as np
import matplotlib.pyplot as plt
from estilo import *
from fluxos import *

N, t = 200, 0.5
x = -1 + (np.arange(N) + 0.5) * (2.0 / N)
casos = [((1.0, -0.5), "choque, $u_L=1$, $u_R=-0.5$ (velocidade $s=0.25$)"),
         ((-1.0, 1.0), "rarefação transônica, $u_L=-1$, $u_R=1$")]
fluxos = [("Godunov", godunov_b, {}, AZUL, "-"),
          ("Roe sem correção", roe_b, {}, VERMELHO, "-"),
          ("Roe + Harten–Hyman", roe_b, {"fix": "hh"}, VERDE, "--"),
          ("Rusanov", rusanov_b, {}, LARANJA, "-")]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
for ax, ((uL, uR), tit) in zip(axs, casos):
    xx = np.linspace(-1, 1, 1001)
    ax.plot(xx, exato_riemann_b(xx, t, uL, uR), "k-", lw=1.0, label="solução de entropia, $t=0.5$")
    u0 = np.where(x < 0, uL, uR)
    for nome, fl, kw, cor, ls in fluxos:
        u, n = avancar_b(u0, x, t, fl, cfl=0.8, **kw)
        ax.plot(x, u, color=cor, ls=ls, lw=1.5, label=nome)
    ax.set_title(tit); ax.set_xlabel("$x$"); ax.set_ylabel("$u$"); ax.set_xlim(-1, 1)
axs[0].legend(loc="upper right"); axs[1].legend(loc="upper left")
axs[0].set_ylim(-0.7, 1.2); axs[1].set_ylim(-1.2, 1.2)
fig.tight_layout(w_pad=2)
salvar(fig, "burgers_godunov_roe.pdf")
