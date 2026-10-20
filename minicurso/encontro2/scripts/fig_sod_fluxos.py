# -*- coding: utf-8 -*-
"""Figura: tubo de Sod com Godunov exato, Roe, HLLC, HLL e Rusanov (N=200)."""
import numpy as np
import matplotlib.pyplot as plt
from estilo import *
from fluxos import *

N, t = 200, 0.2
x, h = malha(0.0, 1.0, N)
xx = np.linspace(0, 1, 2001)
rex, uex, pex = amostrar((xx - 0.5) / t, *SOD[0], *SOD[1])
fluxos = [("Godunov exato", godunov_e, AZUL, "-"), ("Roe + HH", roe_e, VERMELHO, "--"),
          ("HLLC", hllc_e, VERDE, "-"), ("HLL", hll_e, ROXO, "-"), ("Rusanov", rusanov_e, LARANJA, "-")]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax1.plot(xx, rex, "k-", lw=1.0, label="exata"); ax2.plot(xx, rex, "k-", lw=1.0)
res = {}
for nome, fl, cor, ls in fluxos:
    (r, u, p), n = avancar_e(riemann_inicial(x, 0.5, *SOD), x, t, fl, cfl=0.8)
    res[nome] = h * np.sum(np.abs(r - rex[np.searchsorted(xx, x)]))
    ax1.plot(x, r, color=cor, ls=ls, lw=1.4, label=nome)
    ax2.plot(x, r, color=cor, ls=ls, lw=1.6, marker=".", ms=4)
ax1.set_title("Sod, $N=200$, $t=0.2$: densidade"); ax1.set_xlabel("$x$"); ax1.set_ylabel("$\\rho$"); ax1.set_xlim(0, 1)
ax1.legend(loc="upper right")
ax2.set_xlim(0.6, 0.9); ax2.set_ylim(0.1, 0.46); ax2.set_xlabel("$x$")
ax2.set_title("zoom: contato (em $x=0.685$) e choque (em $x=0.850$)")
fig.tight_layout(w_pad=2)
salvar(fig, "sod_fluxos.pdf")
print({k: "%.3e" % v for k, v in res.items()})
