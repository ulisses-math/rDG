# -*- coding: utf-8 -*-
"""Figura 1: 'degrau vezes Dirac'. Três regularizações do mesmo salto e os caminhos no plano de estados."""
import numpy as np
import matplotlib.pyplot as plt
from estilo import *


def rampa(x, centro, eps):
    return np.clip((x - centro) / eps + 0.5, 0.0, 1.0)


eps = 0.2
x = np.linspace(-1, 1, 4001)
casos = [(-0.3, 0.3, "(a) $w$ muda antes de $u$"), (0.0, 0.0, "(b) $u$ e $w$ mudam juntos"), (0.3, -0.3, "(c) $u$ muda antes de $w$")]
cores = [VERDE, LARANJA, AZUL]
fig = plt.figure(figsize=(11.2, 3.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.35)
for k, (cw, cu, tit) in enumerate(casos):
    ax = fig.add_subplot(gs[0, k])
    u = rampa(x, cu, eps); w = rampa(x, cw, eps)
    du = np.gradient(u, x)
    I = np.trapezoid(w * du, x)
    ax.fill_between(x, 0, w * du * eps, color=cores[k], alpha=0.25, lw=0, label=r"$w\,\partial_x u$ (escala $\times\varepsilon$)")
    ax.plot(x, u, color="k", lw=1.6, label="$u$")
    ax.plot(x, w, color=cores[k], lw=1.6, ls="--", label="$w$")
    ax.set_title(tit + "\n" + r"$\int w\,\partial_x u\,\mathrm{d}x=%.2g$" % I, fontsize=9.5)
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.05, 1.25); ax.set_xlabel("$x$")
    if k == 0:
        ax.legend(loc="upper left", fontsize=7.5)
ax = fig.add_subplot(gs[0, 3])
ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="0.6", lw=0.8, ls=":"))
ax.fill_between([0, 1], [0, 1], color=LARANJA, alpha=0.18, lw=0)
ax.plot([0, 0, 1], [0, 1, 1], color=VERDE, lw=2.2, label="(a): área $=1$")
ax.plot([0, 1], [0, 1], color=LARANJA, lw=2.2, label="(b): área $=1/2$")
ax.plot([0, 1, 1], [0, 0, 1], color=AZUL, lw=2.2, label="(c): área $=0$")
ax.plot([0], [0], "ko", ms=5); ax.plot([1], [1], "ko", ms=5)
ax.annotate("$(u_L,w_L)$", (0, 0), xytext=(0.06, -0.14), fontsize=9)
ax.annotate("$(u_R,w_R)$", (1, 1), xytext=(0.62, 1.07), fontsize=9)
ax.set_xlim(-0.12, 1.15); ax.set_ylim(-0.2, 1.25); ax.set_aspect("equal")
ax.set_xlabel("$u$"); ax.set_ylabel("$w$")
ax.set_title("caminhos no plano $(u,w)$:\n" + r"$\int w\,\mathrm{d}u$ = área sob o caminho", fontsize=9.5, pad=8)
ax.legend(loc="center right", fontsize=7.5, bbox_to_anchor=(1.02, 0.42))
salvar(fig, "degrau_dirac.pdf")
