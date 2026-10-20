# -*- coding: utf-8 -*-
"""Figura: problema 123 (Einfeldt): HLL, HLLC e Godunov sobrevivem, Roe gera pressão negativa."""
import numpy as np
import matplotlib.pyplot as plt
from estilo import *
from fluxos import *

N, t = 200, 0.15
x, h = malha(0.0, 1.0, N)
xx = np.linspace(0, 1, 2001)
rex, uex, pex = amostrar((xx - 0.5) / t, *P123[0], *P123[1])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9))
ax1.plot(xx, rex, "k-", lw=1.0, label="exata, $t=0.15$")
ax2.plot(xx, pex, "k-", lw=1.0)
for nome, fl, cor, ls in [("Godunov exato", godunov_e, AZUL, "-"), ("HLLC", hllc_e, VERDE, "--"), ("HLL", hll_e, ROXO, "-")]:
    (r, u, p), n = avancar_e(riemann_inicial(x, 0.5, *P123), x, t, fl, cfl=0.8)
    ax1.plot(x, r, color=cor, ls=ls, lw=1.4, label=nome)
    ax2.plot(x, p, color=cor, ls=ls, lw=1.4)
# Roe: um único passo e o estado que resulta
U = prim2cons(*riemann_inicial(x, 0.5, *P123))
rho, u, p = cons2prim(U)
dt = 0.8 * h / np.max(np.abs(u) + som(rho, p))
Ue = np.concatenate((U[:, :1], U, U[:, -1:]), axis=1)
F = roe_e(Ue[:, :-1], Ue[:, 1:])
U1 = U - dt / h * (F[:, 1:] - F[:, :-1])
r1, u1, p1 = cons2prim(U1)
ax2.plot(x, p1, "s-", color=VERMELHO, ms=3, lw=1.2)
ax1.plot(x, r1, "s-", color=VERMELHO, ms=3, lw=1.2, label="Roe + HH após 1 passo ($\\Delta t=%.4f$)" % dt)
ax2.axhline(0, color=VERMELHO, lw=0.8, ls=":")
ax1.set_title("Problema 123 ($\\rho=1$, $u=\\mp2$, $p=0.4$): densidade"); ax1.set_xlabel("$x$"); ax1.set_ylabel("$\\rho$")
ax2.set_title("pressão: o Roe produz $p<0$ no primeiro passo"); ax2.set_xlabel("$x$"); ax2.set_ylabel("$p$")
ax1.set_xlim(0, 1); ax2.set_xlim(0, 1); ax2.set_ylim(-0.5, 0.5)
h1, l1 = ax1.get_legend_handles_labels()
fig.legend(h1, l1, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), frameon=False)
fig.tight_layout(w_pad=2, rect=(0, 0.02, 1, 1))
salvar(fig, "roe_123.pdf")
print("Roe após 1 passo: min rho = %.4f, min p = %.4f" % (r1.min(), p1.min()))
