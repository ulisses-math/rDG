# -*- coding: utf-8 -*-
"""Figura: a função de pressão f(p) = f_L + f_R + Δu do problema de Sod e a raiz p*."""
import numpy as np
import matplotlib.pyplot as plt
from estilo import *
from riemann_exato import _fK, som, riemann_exato, GAMMA

rL, uL, pL = 1.0, 0.0, 1.0
rR, uR, pR = 0.125, 0.0, 0.1
aL, aR = som(rL, pL), som(rR, pR)
p = np.linspace(0.01, 1.4, 600)
fL, _ = _fK(p, rL, pL, aL, GAMMA)
fR, _ = _fK(p, rR, pR, aR, GAMMA)
ps, us = riemann_exato(rL, uL, pL, rR, uR, pR)
fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.plot(p, fL, color=AZUL, lw=1.6, label="$f_L(p)$: rarefação se $p<p_L$, choque se $p>p_L$")
ax.plot(p, fR, color=VERDE, lw=1.6, label="$f_R(p)$: rarefação se $p<p_R$, choque se $p>p_R$")
ax.plot(p, fL + fR + (uR - uL), color=VERMELHO, lw=2.2, label="$f(p)=f_L+f_R+(u_R-u_L)$")
ax.axhline(0, color="k", lw=0.8)
ax.axvline(pL, color=AZUL, lw=0.8, ls=":"); ax.axvline(pR, color=VERDE, lw=0.8, ls=":")
ax.text(pL + 0.02, -2.4, "$p_L=1$", color=AZUL, fontsize=9)
ax.text(pR + 0.02, -2.4, "$p_R=0.1$", color=VERDE, fontsize=9)
ax.plot([ps], [0], "o", color=LARANJA, ms=8, zorder=5)
ax.annotate("$p^*=%.3f$\n$u^*=%.3f$" % (ps, us), (ps, 0), xytext=(0.55, -1.6), fontsize=10, color=LARANJA,
            arrowprops=dict(arrowstyle="-|>", color=LARANJA))
ax.set_xlim(0, 1.4); ax.set_ylim(-3, 3.2)
ax.set_xlabel("$p$"); ax.set_ylabel("velocidade")
ax.set_title("Sod: $f$ é crescente e côncava, a raiz é única")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
salvar(fig, "euler_pfunc.pdf")
